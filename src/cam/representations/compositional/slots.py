"""Slots architecture: slot attention groups the component tokens, without hand-written composition
(see __init__.py). SlotAttention is also used by the geometric architecture."""

import torch
from torch import nn

from cam.representations.compositional.components import ComponentEmbeddings, mlp
from cam.representations.compositional.structure import NUM_KINDS, OperatorLayout, OperatorParts


class SlotAttention(nn.Module):
    """Slot attention (Locatello et al., 2020) with learned initial slots, so slots are deterministic and
    ordered. Each round, slots compete for the inputs (softmax over slots), take the attention-weighted
    mean of their inputs, and update through a GRU and a residual MLP. Masked inputs are ignored."""

    def __init__(self, dim: int, num_slots: int, iterations: int):
        super().__init__()
        self.initial_slots = nn.Parameter(torch.randn(num_slots, dim) * dim**-0.5)
        self.iterations, self.scale = iterations, dim**-0.5
        self.norm_inputs, self.norm_slots, self.norm_update = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.query, self.key, self.value = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.gru = nn.GRUCell(dim, dim)
        self.update = mlp(dim, dim, 2 * dim)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, N, dim) inputs and (B, N) mask -> (B, num_slots, dim) slots."""
        batch, _, dim = inputs.shape
        inputs = self.norm_inputs(inputs)
        keys, values = self.key(inputs), self.value(inputs)
        slots = self.initial_slots.expand(batch, -1, -1)
        for _ in range(self.iterations):
            logits = torch.einsum("bkd,bnd->bkn", self.query(self.norm_slots(slots)), keys) * self.scale
            attention = logits.softmax(dim=1) * mask[:, None, :]
            attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-8)
            updates = torch.einsum("bkn,bnd->bkd", attention, values)
            slots = self.gru(updates.reshape(-1, dim), slots.reshape(-1, dim)).reshape(batch, -1, dim)
            slots = slots + self.update(self.norm_update(slots))
        return slots


class SlotComposition(nn.Module):
    """Component tokens plus kind embeddings, grouped into num_slots slots, flattened and projected to the
    operator embedding. Invariant to literal order and to EMPTY (padding) positions."""

    def __init__(self, layout: OperatorLayout, component_dim: int, output_dim: int, num_slots: int = 4,
                 slot_iterations: int = 3):
        super().__init__()
        self.components = ComponentEmbeddings(layout, component_dim)
        self.kind = nn.Embedding(NUM_KINDS, component_dim)
        self.slot_attention = SlotAttention(component_dim, num_slots, slot_iterations)
        self.output = nn.Linear(num_slots * component_dim, output_dim)

    def forward(self, parts: OperatorParts) -> torch.Tensor:
        tokens, kinds, mask = self.components.tokens(parts)
        return self.output(self.slot_attention(tokens + self.kind(kinds), mask).flatten(1))
