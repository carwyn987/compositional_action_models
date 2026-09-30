"""Symbol vocabulary shared by every action model format and environment.

Pure domain objects: no dependency on torch, gym, RL, or logging.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TypedParameter:
    name: str  # e.g. "?o"
    type: str  # e.g. "block"


@dataclass(frozen=True)
class Predicate:
    name: str  # e.g. "on"
    args: tuple[str, ...]  # variables ("?o", "?b") or objects ("block0", "block1")
