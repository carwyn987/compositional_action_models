"""Baseline encoder: one entry per training operator, no structure."""

import numpy as np

from cam.domain.symbolic_action_model import SymbolicActionModel
from cam.representations.operator_encoder import OperatorEncoder


class OneHotOperatorEncoder(OperatorEncoder):
    """1 at the operator's position among the given operators (environment_setup gives it every registered
    skill, so indices do not depend on the training skills); any other operator is an error.

    aliases maps an operator name to another whose position it takes (e.g. {"unstack-raised": "unstack"}:
    the repaired operator is encoded exactly like the original)."""

    def __init__(self, models: list[SymbolicActionModel], aliases: dict[str, str] | None = None):
        self.aliases = dict(aliases or {})
        self.index: dict[str, int] = {}
        for model in models:
            self.index.setdefault(self.aliases.get(model.name, model.name), len(self.index))

    @property
    def dim(self) -> int:
        return len(self.index)

    def encode(self, model: SymbolicActionModel) -> np.ndarray:
        encoding = np.zeros(self.dim, dtype=np.float32)
        name = self.aliases.get(model.name, model.name)
        if name in self.index:
            encoding[self.index[name]] = 1.0
        else:
            raise ValueError(f"{model.name} is not one of the training operators {sorted(self.index)}")
        return encoding
