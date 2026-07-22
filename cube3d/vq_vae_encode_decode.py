import argparse
import logging

import numpy as np
import torch
import trimesh

from cube3d.inference.utils import load_config, load_model_weights, parse_structured, select_device
from cube3d.model.autoencoder.one_d_autoencoder import OneDAutoEncoder

MESH_SCALE = 0.96


def apply_transform(vertices: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """Center then uniformly scale a point set: ``(vertices - center) * scale``."""
    return (vertices - center) * scale


def cube_transform(
    vertices: np.ndarray, fill_fraction: float = 1.0, mesh_scale: float = MESH_SCALE
) -> tuple[np.ndarray, float]:
    """``(center, scale)`` fitting ``vertices`` into ``fill_fraction`` of the cube.

    ``center`` is the bounding-box midpoint; ``scale`` maps the box's largest side to
    ``fill_fraction * 2 * mesh_scale``. Concatenate two point sets to frame them jointly;
    ``fill_fraction < 1`` leaves ``1 - fill_fraction`` of headroom for an edited partner.
    """
    bbmin, bbmax = vertices.min(0), vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * fill_fraction * mesh_scale / (bbmax - bbmin).max()
    return center, scale


def rescale(
    vertices: np.ndarray, mesh_scale: float = MESH_SCALE, fill_fraction: float = 1.0
) -> np.ndarray:
    """Rescale vertices into ``fill_fraction`` of the cube (see ``cube_transform``)."""
    return apply_transform(vertices, *cube_transform(vertices, fill_fraction, mesh_scale))


def rescale_pair(
    first: np.ndarray, second: np.ndarray, mesh_scale: float = MESH_SCALE
) -> tuple[np.ndarray, np.ndarray]:
    """Rescale two point sets jointly into the same cube (frame from their union)."""
    center, scale = cube_transform(np.concatenate([first, second]), mesh_scale=mesh_scale)
    return apply_transform(first, center, scale), apply_transform(second, center, scale)


def growth_factor(base: np.ndarray, edited: np.ndarray) -> float:
    """How far ``edited`` reaches beyond ``base``, in units of base's largest side.

    ``reach`` is the farthest ``edited`` point from base's center along any axis;
    returns ``2 * reach / base_side``. 1.0 means no growth, 2.0 a symmetric doubling.
    """
    bbmin, bbmax = base.min(0), base.max(0)
    center = (bbmin + bbmax) * 0.5
    reach = float(np.abs(edited - center).max())
    return 2.0 * reach / float((bbmax - bbmin).max())


def derived_fill(growth: float, margin: float, fill_floor: float) -> float:
    """Fraction of the calculated cube to draw an unedited shape at so its edited partner fits: ``1 / (margin * growth)``.
    """
    return float(min(1.0, max(fill_floor, 1.0 / (margin * max(growth, 1.0)))))


def _load_clean_mesh(file_path: str) -> trimesh.Trimesh:
    """Load a mesh and clean it, without rescaling."""
    mesh: trimesh.Trimesh = trimesh.load(file_path, force="mesh")
    mesh.remove_infinite_values()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Mesh has no vertices or faces after cleaning")
    return mesh


def load_scaled_mesh(file_path: str) -> trimesh.Trimesh:
    """Load a mesh and scale it to a unit cube, and clean the mesh."""
    mesh = _load_clean_mesh(file_path)
    mesh.vertices = rescale(mesh.vertices)
    return mesh


def load_scaled_mesh_pair(
    first_path: str, second_path: str
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Load two meshes and scale them jointly to fit the same cube."""
    first_mesh = _load_clean_mesh(first_path)
    second_mesh = _load_clean_mesh(second_path)
    first_mesh.vertices, second_mesh.vertices = rescale_pair(
        first_mesh.vertices, second_mesh.vertices
    )
    return first_mesh, second_mesh


def _sample_point_cloud(mesh: trimesh.Trimesh, n_samples: int) -> torch.Tensor:
    """Sample points + normals from a mesh surface into a (1, n_samples, 6) tensor."""
    positions, face_indices = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_indices]
    point_cloud = np.concatenate([positions, normals], axis=1)
    return torch.from_numpy(point_cloud.reshape(1, -1, 6)).float()


def sample_scaled_cloud(
    mesh: trimesh.Trimesh, n_samples: int, center: np.ndarray, scale: float
) -> np.ndarray:
    """Sample an ``(n_samples, 6)`` position+normal cloud, positions transformed by
    ``(center, scale)``. Normals are unchanged by the uniform scale and translation."""
    positions, face_indices = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_indices]
    positions = apply_transform(positions, center, scale)
    return np.concatenate([positions, normals], axis=1).astype(np.float32)


def load_and_process_mesh(file_path: str, n_samples: int = 8192) -> torch.Tensor:
    """Loads a 3D mesh, samples points, returns a (1, n_samples, 6) point cloud."""
    mesh = load_scaled_mesh(file_path)
    return _sample_point_cloud(mesh, n_samples)


def load_and_process_mesh_pair(
    first_path: str, second_path: str, n_samples: int = 8192
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads two meshes scaled into the same cube, returns their point clouds."""
    first_mesh, second_mesh = load_scaled_mesh_pair(first_path, second_path)
    return (
        _sample_point_cloud(first_mesh, n_samples),
        _sample_point_cloud(second_mesh, n_samples),
    )


@torch.inference_mode()
def run_shape_decode(
    shape_model: OneDAutoEncoder,
    output_ids: torch.Tensor,
    resolution_base: float = 8.0,
    chunk_size: int = 100_000,
):
    """
    Decodes the shape from the given output IDs and extracts the geometry.
    Args:
        shape_model (OneDAutoEncoder): The shape model.
        output_ids (torch.Tensor): The tensor containing the output IDs.
        resolution_base (float, optional): The base resolution for geometry extraction. Defaults to 8.43.
        chunk_size (int, optional): The chunk size for processing. Defaults to 100,000.
    Returns:
        tuple: A tuple containing the vertices and faces of the mesh.
    """
    shape_ids = (
        output_ids[:, : shape_model.cfg.num_encoder_latents, ...]
        .clamp_(0, shape_model.cfg.num_codes - 1)
        .view(-1, shape_model.cfg.num_encoder_latents)
    )
    latents = shape_model.decode_indices(shape_ids)
    mesh_v_f, _ = shape_model.extract_geometry(
        latents,
        resolution_base=resolution_base,
        chunk_size=chunk_size,
        use_warp=True,
    )
    return mesh_v_f


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="cube shape encode and decode example script"
    )
    parser.add_argument(
        "--mesh-path",
        type=str,
        required=True,
        help="Path to the input mesh file.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="cube3d/configs/open_model.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--shape-ckpt-path",
        type=str,
        required=True,
        help="Path to the shape encoder/decoder checkpoint file.",
    )
    parser.add_argument(
        "--recovered-mesh-path",
        type=str,
        default="recovered_mesh.obj",
        help="Path to save the recovered mesh file.",
    )
    args = parser.parse_args()
    device = select_device()
    logging.info(f"Using device: {device}")

    cfg = load_config(args.config_path)

    shape_model = OneDAutoEncoder(
        parse_structured(OneDAutoEncoder.Config, cfg.shape_model)
    )
    load_model_weights(
        shape_model,
        args.shape_ckpt_path,
    )
    shape_model = shape_model.eval().to(device)
    point_cloud = load_and_process_mesh(args.mesh_path)
    output = shape_model.encode(point_cloud.to(device))
    indices = output[3]["indices"]
    print("Got the following shape indices:")
    print(indices)
    print("Indices shape: ", indices.shape)
    mesh_v_f = run_shape_decode(shape_model, indices)
    vertices, faces = mesh_v_f[0][0], mesh_v_f[0][1]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(args.recovered_mesh_path)
