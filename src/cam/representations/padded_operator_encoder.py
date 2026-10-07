"""Zero-pads another encoder's output to a fixed size, so all embedding methods share one size."""

import numpy as np

from cam.domain.symbolic_action_model import SymbolicActionModel
from cam.representations.operator_encoder import OperatorEncoder


class PaddedOperatorEncoder(OperatorEncoder):
    """The wrapped encoding in the first entries, zeros after it, length dim. Raises ValueError if the
    wrapped encoder is larger than dim."""

    def __init__(self, encoder: OperatorEncoder, dim: int):
        if encoder.dim > dim:
            raise ValueError(f"{type(encoder).__name__} has size {encoder.dim}, larger than the embedding size {dim}")
        self.encoder = encoder
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, model: SymbolicActionModel) -> np.ndarray:
        encoding = np.zeros(self._dim, dtype=np.float32)
        inner = self.encoder.encode(model)
        encoding[: inner.shape[0]] = inner
        return encoding
