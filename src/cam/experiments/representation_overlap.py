"""How much of their operators' structure a set of skills shares: a static check, computed from the PDDL operators
alone (no simulator, no training) in well under a second.

Why it matters. A policy conditioned on the operator's structure can reuse what it learned for one skill on another
only through structure the two share: a literal feature both have (multi-hot), or a component both are composed
from (compositional). Identifier embeddings (one-hot, random) share nothing by construction. So the hypothesis is
that the more of a skill's structure the other training skills share, the more a structured embedding can help it
(RQ1), and that a held-out operator can be carried out zero-shot only from structure seen in training (RQ2, RQ3).
This module measures the sharing; whether the policy uses it is what the experiments measure.

What is counted. Each operator is reduced to a set of units at each of five levels, from coarse to fine:

    predicates  the predicate names it mentions anywhere. pickup: {clear, gripper-empty, holding, on-table, raised}
    atoms       each predicate with its arguments by parameter position, wherever it appears and with either sign:
                clear(0), gripper-empty(), holding(0), on-table(0), raised(0). Variable names do not matter: ?o and ?x
                in the first parameter are both 0, as in the policy's grounding slots
    literals    each atom with its place in the operator and its sign: precondition on-table(0), add holding(0),
                delete raised(0); preconditions are "precondition" or "negated precondition", effects "add" or
                "delete". These are exactly the positional features of the multi-hot embedding (pickup has 8) and the
                literal components the compositional embeddings compose
    sections    the whole precondition, and the whole effect, as one unit each: two skills share one only if they have
                the same set of preconditions (or of effects)
    operators   the whole operator, preconditions and effects (the name is left out): shared only by identical ones

Coarse levels are shared more than fine ones: most blocksworld skills mention clear or holding, fewer share a
literal in the same place with the same sign, few share a whole precondition, and no two skills have the same
operator (otherwise they would be the same skill to a planner).

Which level each embedding reuses. One-hot and random identifiers share nothing at any level. Multi-hot shares
literals only: its positional features are exactly the literals level (plus a coarser feature per section and
predicate, without arguments), so the literals row is its similarity, and to it precondition on-table(0) and add
on-table(0) are unrelated. The compositional encoders reuse three levels at once: each predicate has one learned
embedding (predicates), an atom gets the same vector wherever it appears, in either section and with either sign
(atoms), and a literal contributes the same term to its section's sum (literals: the tree's AND and PRE/EFF, the slot
and grid tokens). So for them the literals row is a lower bound, and the atoms and predicates rows are what they link
on top. The sections row understates them: it counts a precondition or effect as shared only when the whole set is
identical, but the tree's AND sums its literals before its final network, so sections that share most literals
look alike to it, which the literals row's pair overlap reflects better. None of this says how similar the trained
embeddings are; that is learned (review_runs.py section 6, and the RQ5 probes).

The measures, at each level, over the skills given (the training skills):
    units         how many distinct units the skills have together
    shared units  the fraction of those units that at least two skills have
    skill shared  per skill, the fraction of its own units that at least one other skill also has, and the mean over
                  the skills. 1: everything the skill is made of is also trained in another skill; 0: nothing is, and
                  an identifier would serve it as well as its structure
    pair overlap  the mean over pairs of skills of their Jaccard similarity, |A ∩ B| / |A ∪ B| of their unit sets
and, for each held-out operator (a repair's TO, or a new skill), its coverage: the fraction of its units that some
training skill has. Coverage 1 at the literal level is what the overnight runner requires of a repair
(check_repairs): every literal of the repaired operator is trained in some other skill, only the combination is new.
"""

from dataclasses import dataclass
from itertools import combinations

from cam.skills.registry import build_skill

LEVELS = ("predicates", "atoms", "literals", "sections", "operators")
# The operator embeddings that reuse each level's units (module docstring: which level each embedding reuses)
REUSED_BY = {
    "predicates": "compositional",
    "atoms": "compositional",
    "literals": "multi-hot (its similarity); compositional (a lower bound)",
    "sections": "compositional, graded (this row: identical sets only)",
    "operators": "none",
}


def operator_units(skill: str) -> dict[str, frozenset[str]]:
    """The skill's operator as a set of units at each level (LEVELS; see the module docstring)."""
    model = build_skill(skill, {"symbolic_action_model_format": "pddl"}).symbolic_action_model
    position = {parameter.name: index for index, parameter in enumerate(model.parameters)}

    def atom(predicate) -> str:
        return f"{predicate.name}({','.join(str(position[argument]) for argument in predicate.args)})"

    preconditions = frozenset(f"{'negated precondition' if p.negated else 'precondition'} {atom(p.predicate)}"
                              for p in model.preconditions)
    effects = frozenset(f"{'delete' if e.delete else 'add'} {atom(e.predicate)}" for e in model.effects)
    predicates = [p.predicate for p in model.preconditions] + [e.predicate for e in model.effects]
    precondition_section = "preconditions {" + ", ".join(sorted(preconditions)) + "}"
    effect_section = "effects {" + ", ".join(sorted(effects)) + "}"
    return {
        "predicates": frozenset(predicate.name for predicate in predicates),
        "atoms": frozenset(atom(predicate) for predicate in predicates),
        "literals": preconditions | effects,
        "sections": frozenset({precondition_section, effect_section}),
        "operators": frozenset({precondition_section + " " + effect_section}),
    }


@dataclass(frozen=True)
class LevelOverlap:
    level: str
    units: int  # distinct units over the skills
    shared_units: float  # fraction of them that at least two skills have
    skill_shared: dict[str, float]  # per skill: fraction of its units another skill also has
    pair_overlap: float  # mean Jaccard similarity over pairs of skills (0 with fewer than two skills)
    coverage: dict[str, float]  # per held-out operator: fraction of its units some skill has

    @property
    def mean_skill_shared(self) -> float:
        return sum(self.skill_shared.values()) / len(self.skill_shared) if self.skill_shared else 0.0


def overlap(skills, held_out=()) -> list[LevelOverlap]:
    """The overlap measures (module docstring) at every level, for the skills and the held-out operators."""
    units = {skill: operator_units(skill) for skill in dict.fromkeys([*skills, *held_out])}
    skills = list(dict.fromkeys(skills))
    result = []
    for level in LEVELS:
        sets = {skill: units[skill][level] for skill in skills}
        everything = set().union(*sets.values()) if sets else set()
        count = {unit: sum(unit in own for own in sets.values()) for unit in everything}
        pairs = list(combinations(skills, 2))
        result.append(LevelOverlap(
            level=level,
            units=len(everything),
            shared_units=sum(n >= 2 for n in count.values()) / len(everything) if everything else 0.0,
            skill_shared={skill: sum(count[unit] >= 2 for unit in own) / len(own) for skill, own in sets.items()},
            pair_overlap=(sum(len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a, b in pairs) / len(pairs)
                          if pairs else 0.0),
            coverage={skill: len(units[skill][level] & everything) / len(units[skill][level]) for skill in held_out},
        ))
    return result


def report(skills, held_out=()) -> str:
    """The overlap of the skills' operators (and the coverage of the held-out ones) as a text table."""
    levels = overlap(skills, held_out)
    title = f"Representation overlap: {len(list(dict.fromkeys(skills)))} training skills"
    lines = [title + (f"; held out: {', '.join(held_out)}" if held_out else "")]
    header = f"  {'level':11s} {'units':>5s}  {'shared units':>12s}  {'skill shared':>12s}  {'pair overlap':>12s}"
    lines.append(header + "".join(f"  {'coverage ' + skill:>{max(14, len(skill) + 9)}s}" for skill in held_out)
                 + "  reused by")
    for row in levels:
        lines.append(f"  {row.level:11s} {row.units:5d}  {row.shared_units:12.2f}  {row.mean_skill_shared:12.2f}  "
                     f"{row.pair_overlap:12.2f}"
                     + "".join(f"  {row.coverage[skill]:>{max(14, len(skill) + 9)}.2f}" for skill in held_out)
                     + f"  {REUSED_BY[row.level]}")
    literal_level = next(row for row in levels if row.level == "literals")
    lines.append("  skill shared, literals: " + ", ".join(f"{skill} {value:.2f}"
                                                          for skill, value in literal_level.skill_shared.items()))
    lines.append("  how to read it: multi-hot shares literals only, so the literals row is its similarity; the compositional "
                 "encoders reuse predicates, atoms and literals at once, so for them the literals row is a lower bound "
                 "and the atoms and predicates rows are what they link on top; identifiers (one-hot, random) share "
                 "nothing")
    return "\n".join(lines)
