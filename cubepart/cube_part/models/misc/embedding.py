import torch
import torch.nn as nn


class SlotEmbedding(nn.Module):
    NUM_SHAPES = 4

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.pair = nn.Parameter(torch.zeros(2, hidden_size)) # (A,A') or (B,B')
        self.state = nn.Parameter(torch.zeros(2, hidden_size)) # edited (A' and B') or unchanged (A and B)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[0] % self.NUM_SHAPES != 0:
            raise ValueError("SlotEmbedding expects four flattened shape slots.")

        slot_embeddings = torch.stack(
            [
                self.pair[0] + self.state[0],
                self.pair[0] + self.state[1],
                self.pair[1] + self.state[0],
                self.pair[1] + self.state[1],
            ]
        )
        batch_size = hidden_states.shape[0] // self.NUM_SHAPES
        slot_embeddings = slot_embeddings.repeat(batch_size, 1)
        slot_embeddings = slot_embeddings.to(hidden_states.dtype).unsqueeze(1)
        return hidden_states + slot_embeddings
