"""Value object describing a symbolic action model to be embedded.

This is intentionally standalone (no dependency on ``fetch_blockworld``) so the
embedding subsystem stays decoupled from the environment package. Callers build
one of these from whatever symbolic source they have (e.g. a ``SkillSpec``).

Today only ``name`` and ``description`` (the full action-model string) are used.
The remaining fields are placeholders for the planned component-wise embedding
scheme (types, parameters, predicates, preconditions, effects) and are safe to
leave empty until those embedders exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Selectable text sources for a text/sentence embedder.
EMBED_SOURCES = ("name", "action_model")


@dataclass(frozen=True)
class SymbolicActionModel:
    """A symbolic action/operator, in a form an embedder can consume."""

    name: str
    # Full action-model string (PDDL-style operator). Used for the
    # sentence/paragraph embedding source. Falls back to ``name`` when absent.
    description: str | None = None

    # --- Placeholders for the future component-wise embedding scheme. ---
    types: tuple[str, ...] = field(default_factory=tuple)
    parameters: tuple[str, ...] = field(default_factory=tuple)
    preconditions: tuple[str, ...] = field(default_factory=tuple)
    effects: tuple[str, ...] = field(default_factory=tuple)

    def text(self, source: str) -> str:
        """Return the text to embed for a given ``source``.

        ``name``         -> the operator name alone.
        ``action_model`` -> the full action-model string (``description``),
                            falling back to the name when no string was given.
        """
        if source == "name":
            return self.name
        if source == "action_model":
            return self.description or self.name
        raise ValueError(
            f"Unknown embed source {source!r}. Choices: {list(EMBED_SOURCES)}"
        )
