"""Baseline encoder: one entry per training operator, no structure."""

import numpy as np

from cam.domain.symbolic_action_model import SymbolicActionModel
from cam.representations.operator_encoder import OperatorEncoder


class OneHotOperatorEncoder(OperatorEncoder):
    """1 at the operator's position among the training operators; any other operator is an error."""

    def __init__(self, models: list[SymbolicActionModel]):
        self.index = {model.name: i for i, model in enumerate(models)}

    @property
    def dim(self) -> int:
        return len(self.index)

    def encode(self, model: SymbolicActionModel) -> np.ndarray:
        encoding = np.zeros(self.dim, dtype=np.float32)
        if model.name in self.index:
            encoding[self.index[model.name]] = 1.0
        else:
            raise ValueError(f"{model.name} is not one of the training operators {sorted(self.index)}")
        return encoding
