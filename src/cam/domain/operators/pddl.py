"""PDDL operator data types: Predicate, TypedParameter, Precondition, Effect, PDDLOperator.

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
    args: tuple[str, ...]  # e.g. ("?o", "?b")


@dataclass(frozen=True)
class Precondition:
    predicate: Predicate
    negated: bool = False  # True for (not ...) preconditions


@dataclass(frozen=True)
class Effect:
    predicate: Predicate
    delete: bool = False  # True for (not ...) effects


@dataclass(frozen=True)
class PDDLOperator:
    name: str
    parameters: tuple[TypedParameter, ...]
    preconditions: tuple[Precondition, ...]
    effects: tuple[Effect, ...]
