"""Write a PDDLOperator back to a canonical PDDL `(:action ...)` string.

Canonical form: one line, single spaces, :precondition and :effect always
wrapped in (and ...), and every parameter typed individually. Equal operators
therefore always produce identical strings, and to_pddl(parse_operator(s)) == s
for any canonical s.
"""

from cam.domain.operators.pddl import (
    Effect,
    PDDLOperator,
    Precondition,
    Predicate,
    TypedParameter,
)


def to_pddl(op: PDDLOperator) -> str:
    """Render op as "(:action name :parameters (...) :precondition (and ...) :effect (and ...))"."""
    parameters = " ".join(_write_parameter(p) for p in op.parameters)
    preconditions = [_write_precondition(p) for p in op.preconditions]
    effects = [_write_effect(e) for e in op.effects]
    return (
        f"(:action {op.name}"
        f" :parameters ({parameters})"
        f" :precondition {_write_and(preconditions)}"
        f" :effect {_write_and(effects)})"
    )


def _write_parameter(param: TypedParameter) -> str:
    return f"{param.name} - {param.type}"


def _write_predicate(pred: Predicate) -> str:
    return "(" + " ".join((pred.name, *pred.args)) + ")"


def _write_precondition(pre: Precondition) -> str:
    return _write_not(pre.predicate) if pre.negated else _write_predicate(pre.predicate)


def _write_effect(eff: Effect) -> str:
    return _write_not(eff.predicate) if eff.delete else _write_predicate(eff.predicate)


def _write_not(pred: Predicate) -> str:
    return f"(not {_write_predicate(pred)})"


def _write_and(literals: list[str]) -> str:
    return "(" + " ".join(("and", *literals)) + ")"
