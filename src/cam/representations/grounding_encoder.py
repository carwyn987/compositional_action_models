"""Grounding encoder: the objects bound to an action model's parameters -> a fixed-size vector.

One slot per parameter, in parameter order, padded to max_arity slots:
    [ used (1) | type one-hot (n_types) | object index one-hot (max_objects) | object features (feature_dim) ]
See docs/policy_inputs.md.
"""

import numpy as np

from cam.domain.symbolic_action_model import GroundedSymbolicActionModel


class GroundingEncoder:
    def __init__(self, object_types: list[str], max_objects: int, feature_dim: int, max_arity: int):
        self.type_index = {object_type: i for i, object_type in enumerate(object_types)}
        self.max_objects = max_objects
        self.feature_dim = feature_dim
        self.max_arity = max_arity
        self.slot_dim = 1 + len(object_types) + max_objects + feature_dim

    @property
    def dim(self) -> int:
        return self.max_arity * self.slot_dim

    def encode(
        self,
        grounded: GroundedSymbolicActionModel,
        objects: dict[str, str],
        object_features: dict[str, np.ndarray],
    ) -> np.ndarray:
        """objects: name -> type, in scene order; object_features: name -> feature vector."""
        if len(grounded.arguments) > self.max_arity:
            raise ValueError(f"{grounded} has more than max_arity={self.max_arity} arguments")
        if len(objects) > self.max_objects:
            raise ValueError(f"{len(objects)} objects exceed max_objects={self.max_objects}")
        object_index = {name: i for i, name in enumerate(objects)}
        encoding = np.zeros((self.max_arity, self.slot_dim), dtype=np.float32)
        for slot, obj in enumerate(grounded.arguments):
            features = np.asarray(object_features[obj], dtype=np.float32)
            if features.shape[0] > self.feature_dim:
                raise ValueError(f"{obj} has {features.shape[0]} features, more than feature_dim={self.feature_dim}")
            offset = 1 + len(self.type_index)
            encoding[slot, 0] = 1.0
            encoding[slot, 1 + self.type_index[objects[obj]]] = 1.0
            encoding[slot, offset + object_index[obj]] = 1.0
            encoding[slot, offset + self.max_objects : offset + self.max_objects + features.shape[0]] = features
        return encoding.reshape(-1)
