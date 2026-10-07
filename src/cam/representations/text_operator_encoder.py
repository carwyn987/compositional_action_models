"""Text embedding of a whole operator: the operator as one text, embedded by a text backend.

The operator's text is either its canonical PDDL (pddl_writer.to_pddl) or a templated paragraph
(operator_paragraph). The text depends on the operator's content, so a repaired operator gets a new
embedding, and any operator (seen or not) can be encoded.
"""

import numpy as np

from cam.domain.pddl.pddl import PDDLOperator
from cam.domain.pddl.pddl_writer import to_pddl
from cam.representations.operator_encoder import OperatorEncoder
from cam.representations.text_backends import TextEmbeddingBackend


def operator_paragraph(operator: PDDLOperator) -> str:
    """A short paragraph describing the operator, e.g.

    Action pickup with parameters ?o (block). It can be applied when (on-table ?o), (clear ?o) and
    (gripper-empty) hold. Afterwards (holding ?o) holds, and (on-table ?o), (clear ?o) and
    (gripper-empty) no longer hold.
    """

    def listing(items: list[str]) -> str:
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

    parameters = ", ".join(f"{param.name} ({param.type})" for param in operator.parameters) or "none"
    required = [str(p.predicate) for p in operator.preconditions if not p.negated]
    forbidden = [str(p.predicate) for p in operator.preconditions if p.negated]
    added = [str(e.predicate) for e in operator.effects if not e.delete]
    deleted = [str(e.predicate) for e in operator.effects if e.delete]

    sentences = [f"Action {operator.name} with parameters {parameters}."]
    conditions = []
    if required:
        conditions.append(f"{listing(required)} {'hold' if len(required) > 1 else 'holds'}")
    if forbidden:
        conditions.append(f"{listing(forbidden)} {'do' if len(forbidden) > 1 else 'does'} not hold")
    sentences.append(f"It can be applied when {' and '.join(conditions)}." if conditions else "It can always be applied.")
    outcomes = []
    if added:
        outcomes.append(f"{listing(added)} {'hold' if len(added) > 1 else 'holds'}")
    if deleted:
        outcomes.append(f"{listing(deleted)} no longer {'hold' if len(deleted) > 1 else 'holds'}")
    if outcomes:
        sentences.append(f"Afterwards {', and '.join(outcomes)}.")
    return " ".join(sentences)


OPERATOR_TEXTS = {"pddl": to_pddl, "paragraph": operator_paragraph}


class TextOperatorEncoder(OperatorEncoder):
    def __init__(self, backend: TextEmbeddingBackend, text: str = "pddl"):
        self.backend = backend
        self.to_text = OPERATOR_TEXTS[text]

    @property
    def dim(self) -> int:
        return self.backend.dim

    def encode(self, operator: PDDLOperator) -> np.ndarray:
        return self.backend.embed_text(self.to_text(operator))
