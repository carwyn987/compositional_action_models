"""Fixed random vector per operator: an arbitrary identifier with no structure (RQ1 baseline)."""

import hashlib

import numpy as np

from cam.domain.symbolic_action_model import SymbolicActionModel
from cam.representations.operator_encoder import OperatorEncoder


class RandomOperatorEncoder(OperatorEncoder):
    """Each operator name maps to a fixed random unit vector of length dim.

    The vector is drawn from a standard normal and normalised, using a random generator
    seeded from the operator name and `seed` (via SHA-256, so it is the same in every
    process and run). Any operator, including one not seen in training, gets its own
    vector; vectors carry no information about the operator's structure. Operators are
    identified by name, so a changed operator that keeps its name keeps its vector. aliases maps an
    operator name to another whose vector it uses (e.g. {"unstack-raised": "unstack"}).
    """

    def __init__(self, dim: int, seed: int = 0, aliases: dict[str, str] | None = None):
        self._dim = dim
        self.seed = seed
        self.aliases = dict(aliases or {})

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, model: SymbolicActionModel) -> np.ndarray:
        name = self.aliases.get(model.name, model.name)
        digest = hashlib.sha256(f"{self.seed}:{name}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        vector = rng.standard_normal(self._dim)
        return (vector / np.linalg.norm(vector)).astype(np.float32)
