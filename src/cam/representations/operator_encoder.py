"""Operator encoders: a lifted symbolic action model -> a fixed-size vector.

Each implementation is one embedding method:
    OneHotOperatorEncoder         one entry per training operator; no structure (baseline)
    RandomOperatorEncoder         fixed random unit vector per operator name; no structure (baseline)
    TextOperatorEncoder           the whole operator as text (PDDL or a paragraph), embedded by a text backend
    PDDLMultiHotOperatorEncoder   one entry per literal feature; structure, nothing learned
Learned methods (embed each literal, then aggregate by mean, attention, slots,
...) implement the same interface. An encoder is built from the training
operators, which fix its vocabulary and output size. See docs/policy_inputs.md.
"""

from abc import ABC, abstractmethod

import numpy as np

from cam.domain.symbolic_action_model import SymbolicActionModel


class OperatorEncoder(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        """Length of every encoding."""

    @abstractmethod
    def encode(self, model: SymbolicActionModel) -> np.ndarray:
        """float32 vector of length dim."""
