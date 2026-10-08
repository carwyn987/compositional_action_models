"""Geometric compositional embedding (experimental): an operator's components laid out on a grid, read by
a CNN and slot attention. One of the --compositional-architecture options; see compositional_embedding.py
for the components, the structure array and the extractor.

The grid is an image with component_dim channels, height H = [1 +] P + L and width W = NUM_KINDS:
    row r      the r-th component in canonical order: [name], parameters by position, literals as
               serialized (sorted by section, predicate, arguments)
    column     the component's kind: name, parameter, precondition, negated precondition, add, delete
    channels   the component's embedding (ComponentEmbeddings.tokens); the row's other cells are zero
so the operator's hierarchy runs down the rows, kind information across the columns, and the
components' content along the channels.

Pipeline, in Slot Attention's standard order (a CNN encoder, then slots): two 3x3 convolutions extract
local patterns across neighbouring components and kinds, a learned positional embedding marks each
cell, slot attention groups the H x W cells into num_slots slots, and the flattened slots are projected
to the operator embedding. Unlike the tree and slots architectures, the result depends on component
order (canonical here) and on the layout's maxima, which set the grid size.
"""

import torch
from torch import nn

from cam.representations.compositional_embedding import (
    NUM_KINDS,
    ComponentEmbeddings,
    OperatorLayout,
    OperatorParts,
    SlotAttention,
)


class GeometricComposition(nn.Module):
    def __init__(self, layout: OperatorLayout, component_dim: int, output_dim: int, num_slots: int = 4,
                 slot_iterations: int = 3):
        super().__init__()
        height = (1 if layout.name_dim else 0) + layout.max_parameters + layout.max_literals
        self.components = ComponentEmbeddings(layout, component_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(component_dim, component_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(component_dim, component_dim, 3, padding=1), nn.ReLU(),
        )
        self.position = nn.Parameter(torch.randn(height * NUM_KINDS, component_dim) * 0.02)
        self.slot_attention = SlotAttention(component_dim, num_slots, slot_iterations)
        self.output = nn.Linear(num_slots * component_dim, output_dim)

    def grid(self, parts: OperatorParts) -> torch.Tensor:
        """(B, component_dim, H, NUM_KINDS): each component's token in its row, in its kind's column."""
        tokens, kinds, mask = self.components.tokens(parts)
        cells = nn.functional.one_hot(kinds, NUM_KINDS).float() * mask[..., None]  # (B, H, W)
        return (cells[..., None] * tokens[:, :, None, :]).permute(0, 3, 1, 2)

    def forward(self, parts: OperatorParts) -> torch.Tensor:
        features = self.cnn(self.grid(parts)).flatten(2).transpose(1, 2) + self.position  # (B, H * W, dim)
        every_cell = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        return self.output(self.slot_attention(features, every_cell).flatten(1))
