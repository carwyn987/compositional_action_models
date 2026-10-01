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

    def __str__(self) -> str:
        return f"({' '.join((self.name, *self.args))})"


def format_facts(facts) -> str:
    """Sorted, space-separated facts, e.g. "(clear block0) (gripper-empty)"; "-" when empty."""
    return " ".join(sorted(str(fact) for fact in facts)) or "-"
