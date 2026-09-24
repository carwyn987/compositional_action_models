"""Parse a single PDDL `(:action ...)` string into a PDDLOperator.

Supported subset: typed parameters, and a conjunction (or a single literal) of
positive / `(not ...)` literals for both :precondition and :effect.
"""

from cam.domain.operators.pddl import (
    Effect,
    PDDLOperator,
    Precondition,
    Predicate,
    TypedParameter,
)

SExpr = str | list["SExpr"]

_ACTION_FIELDS = (":parameters", ":precondition", ":effect")


def tokenize(text: str) -> list[str]:
    """Split PDDL text into parens and symbols, e.g. "(on ?o)" -> ["(", "on", "?o", ")"]."""
    return text.replace("(", " ( ").replace(")", " ) ").split()


def parse_sexpr(text: str) -> SExpr:
    """Read exactly one s-expression, e.g. "(a (b ?x))" -> ["a", ["b", "?x"]]."""
    tokens = tokenize(text)
    if not tokens:
        raise ValueError("empty input")
    expr, pos = _read(tokens, 0)
    if pos != len(tokens):
        raise ValueError(f"unexpected tokens after expression: {tokens[pos:]}")
    return expr


def _read(tokens: list[str], pos: int) -> tuple[SExpr, int]:
    """Read one expression starting at tokens[pos]; return it and the next position."""
    token = tokens[pos]
    if token == ")":
        raise ValueError("unexpected ')'")
    if token != "(":
        return token, pos + 1

    items: list[SExpr] = []
    pos += 1
    while pos < len(tokens) and tokens[pos] != ")":
        item, pos = _read(tokens, pos)
        items.append(item)
    if pos == len(tokens):
        raise ValueError("missing ')'")
    return items, pos + 1


def parse_operator(text: str) -> PDDLOperator:
    """Parse "(:action name :parameters (...) :precondition (...) :effect (...))"."""
    expr = parse_sexpr(text)
    if not isinstance(expr, list) or len(expr) < 2 or expr[0] != ":action":
        raise ValueError("expected (:action <name> ...)")
    name = expr[1]
    if not isinstance(name, str):
        raise ValueError(f"action name must be a symbol, got {name!r}")

    fields = _parse_fields(expr[2:])
    return PDDLOperator(
        name=name,
        parameters=_parse_parameters(fields[":parameters"]),
        preconditions=tuple(
            Precondition(pred, negated)
            for pred, negated in _parse_conjunction(fields[":precondition"])
        ),
        effects=tuple(
            Effect(pred, delete)
            for pred, delete in _parse_conjunction(fields[":effect"])
        ),
    )


def _parse_fields(items: list[SExpr]) -> dict[str, SExpr]:
    """Pair up ":keyword value" items; require exactly the known action fields."""
    if len(items) % 2 != 0:
        raise ValueError("expected :keyword value pairs")
    fields: dict[str, SExpr] = {}
    for key, value in zip(items[::2], items[1::2]):
        if key not in _ACTION_FIELDS:
            raise ValueError(f"unknown action field {key!r}")
        if key in fields:
            raise ValueError(f"duplicate action field {key!r}")
        fields[key] = value
    missing = [key for key in _ACTION_FIELDS if key not in fields]
    if missing:
        raise ValueError(f"missing action fields: {missing}")
    return fields


def _parse_parameters(expr: SExpr) -> tuple[TypedParameter, ...]:
    """Parse "(?a ?b - block ?c - arm)"; every variable must be typed."""
    if not isinstance(expr, list):
        raise ValueError(f"expected parameter list, got {expr!r}")
    params: list[TypedParameter] = []
    pending: list[str] = []  # variables waiting for their "- type"
    tokens = iter(expr)
    for token in tokens:
        if token == "-":
            type_name = next(tokens, None)
            if not pending or not isinstance(type_name, str):
                raise ValueError(f"malformed typed parameter list: {expr!r}")
            params.extend(TypedParameter(var, type_name) for var in pending)
            pending = []
        elif isinstance(token, str) and token.startswith("?"):
            pending.append(token)
        else:
            raise ValueError(f"unexpected token in parameters: {token!r}")
    if pending:
        raise ValueError(f"untyped parameters: {pending}")
    return tuple(params)


def _parse_conjunction(expr: SExpr) -> list[tuple[Predicate, bool]]:
    """Parse "(and lit ...)" or a single literal into (predicate, negated) pairs."""
    if isinstance(expr, list) and expr and expr[0] == "and":
        return [_parse_literal(lit) for lit in expr[1:]]
    return [_parse_literal(expr)]


def _parse_literal(expr: SExpr) -> tuple[Predicate, bool]:
    """Parse "(p ?x)" -> (p, False) or "(not (p ?x))" -> (p, True)."""
    if isinstance(expr, list) and expr and expr[0] == "not":
        if len(expr) != 2:
            raise ValueError(f"(not ...) takes exactly one literal: {expr!r}")
        return _parse_predicate(expr[1]), True
    return _parse_predicate(expr), False


def _parse_predicate(expr: SExpr) -> Predicate:
    """Parse "(name arg ...)" where all parts are symbols."""
    if not isinstance(expr, list) or not expr or not all(isinstance(t, str) for t in expr):
        raise ValueError(f"expected atomic predicate, got {expr!r}")
    if expr[0] in ("and", "not"):
        raise ValueError(f"nested {expr[0]!r} is not supported: {expr!r}")
    return Predicate(expr[0], tuple(expr[1:]))
