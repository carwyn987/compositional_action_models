"""What the RQ5 embedding explorer (scripts/explore_embeddings.py) plots: operator embeddings, and the parts they are
built from, as full-dimensional vectors that know what they are.

Sources. A source is one embedding function, built the way training builds it (OPERATOR_ENCODERS from a config:
main.py's defaults, or a run's config.json), so a run is explored exactly as its policy saw it:
    FixedEncoderSource   an encoder with nothing learned: one-hot, random, multi-hot or text
    TrainableRunSource   a --trainable-operator-embedding run: the learned matrix W of one of its policy's features
                         extractors, applied to the encoder's vector x
    CompositionalSource  a compositional features extractor, a run's or an untrained one (the same architecture and
                         settings, freshly initialized: the control for the structure a composition has by
                         construction): its component embeddings and every stage of its composition

Levels. Each source has levels (Level): the operator embedding the policy sees and, below it, whatever the source is
built from, down to literals, predicates, variables and types. Two levels of a source share a space only if they are
inputs to the same function, or one is a linear combination of the other (the tree's and-sum, a sum of conjuncts,
shares the conjuncts' space); only points in one space are compared, projected together or added up. The same
dimension is not enough: the tree's precondition and effect embeddings have the same size but come out of
different functions.

Operators. Points are computed for the registered operators (SKILL_REGISTRY) and for custom PDDL, each with a label
of its own: the name is an input to one-hot, random, text and compositional-with-name embeddings, so a custom
operator that keeps its original's name and changes a literal isolates the change of content. Component points
(literals, atoms, features, ...) are collected from the operators, once per distinct component: variables count by
parameter position, as every encoder sees them, and are named as in the first operator that has the component.
Anything an encoder cannot encode (an unknown name for one-hot, a predicate outside the layout, a text missing from
the cache with --offline) is reported with the encoder's reason instead of a point.
"""

import re
from dataclasses import dataclass
from itertools import permutations

import numpy as np

from cam.domain.pddl.pddl import Effect, PDDLOperator, Precondition
from cam.domain.pddl.pddl_parser import parse_operator, tokenize
from cam.domain.pddl.pddl_writer import to_pddl
from cam.domain.symbols import Predicate
from cam.representations.pddl_multi_hot_operator_encoder import SECTIONS, operator_features
from cam.representations.text_backends import TextEmbeddingBackend


@dataclass(frozen=True)
class Level:
    name: str  # e.g. operator, operator-raw, literal, atom, and-sum, slot
    kind: str  # operator | section | slot | name | literal | atom | feature | predicate | type | variable | typed-variable
    space: str  # levels of one source with the same space are projected together and may be added up
    dim: int
    description: str


@dataclass(frozen=True, eq=False)
class EmbeddingPoint:
    label: str
    kind: str
    level: str
    space: str
    source: str
    vector: np.ndarray  # full-dimensional
    origin: str  # registered | custom | probe | vocabulary (components and vocabulary entries) | expression
    operators: tuple[str, ...]  # the operators the point belongs to (an operator's own label for operator points)
    trained: bool | None  # run sources: whether one of the operators is among the run's skills; static: None
    text: str = ""  # hover details


@dataclass(frozen=True)
class LabelledOperator:
    label: str
    operator: PDDLOperator
    origin: str = "registered"  # registered | custom | probe


def registered_operators(symbolic_action_model_format: str = "pddl") -> list[LabelledOperator]:
    """Every registered skill's operator, in registry order."""
    from cam.domain.action_model_library.loader import load_symbolic_action_model
    from cam.skills.registry import SKILL_REGISTRY

    return [LabelledOperator(name, load_symbolic_action_model(name, symbolic_action_model_format))
            for name in SKILL_REGISTRY]


def parse_custom(label: str, pddl: str, taken: set[str]) -> PDDLOperator:
    """A custom point's operator; ValueError (with the parser's message) for invalid PDDL or a label in use."""
    if not label.strip():
        raise ValueError("a custom point needs a label")
    if label in taken:
        raise ValueError(f"the label {label!r} is already used")
    return parse_operator(pddl)


def literal_label(section: str, predicate: Predicate, negated: bool) -> str:
    """e.g. "pre (clear ?o)", "eff (not (raised ?o))": a literal and where it is (pre: precondition, eff: effect)."""
    return f"{section} (not {predicate})" if negated else f"{section} {predicate}"


def operator_literals(operator: PDDLOperator) -> list[tuple[str, Predicate, bool]]:
    """(section, predicate, negated) of each precondition and effect (a delete effect is a negated effect)."""
    return ([("pre", p.predicate, p.negated) for p in operator.preconditions]
            + [("eff", e.predicate, e.delete) for e in operator.effects])


def one_literal_operator(operator: PDDLOperator, section: str, predicate: Predicate, negated: bool) -> PDDLOperator:
    """The operator with only this literal: its name and parameters, and one precondition or effect."""
    if section == "pre":
        return PDDLOperator(operator.name, operator.parameters, (Precondition(predicate, negated),), ())
    return PDDLOperator(operator.name, operator.parameters, (), (Effect(predicate, negated),))


def positions(operator: PDDLOperator, predicate: Predicate) -> tuple:
    index = {parameter.name: i for i, parameter in enumerate(operator.parameters)}
    return tuple(index.get(argument, argument) for argument in predicate.args)


def literal_components(operators: list[LabelledOperator]) -> dict[tuple, tuple[str, PDDLOperator, list[str]]]:
    """Each distinct literal (by section, sign, predicate and argument positions) -> (its label, its one-literal
    operator on the first operator that has it, the labels of every operator that has it)."""
    components: dict[tuple, tuple[str, PDDLOperator, list[str]]] = {}
    for entry in operators:
        for section, predicate, negated in operator_literals(entry.operator):
            key = (section, negated, predicate.name, positions(entry.operator, predicate))
            components.setdefault(key, (literal_label(section, predicate, negated),
                                        one_literal_operator(entry.operator, section, predicate, negated), []))
            components[key][2].append(entry.label)
    return components


def literal_jaccard(a: PDDLOperator, b: PDDLOperator, sections=SECTIONS) -> float:
    """The structural reference: Jaccard similarity of the operators' literals (multi-hot's positional features:
    section and sign, predicate, argument positions), of the given sections only (e.g. "add" and "delete")."""
    first = {feature for feature in operator_features(a) if len(feature) == 3 and feature[0] in sections}
    second = {feature for feature in operator_features(b) if len(feature) == 3 and feature[0] in sections}
    return len(first & second) / len(first | second) if first | second else 1.0


def multi_hot_cosine(a: PDDLOperator, b: PDDLOperator) -> float:
    """Cosine similarity of the operators' multi-hot vectors (every feature, positional and name-only)."""
    first, second = operator_features(a), operator_features(b)
    return len(first & second) / np.sqrt(len(first) * len(second)) if first and second else 0.0


class CacheOnlyTextBackend(TextEmbeddingBackend):
    """--offline: stands in for a cached backend's API backend, so a text missing from the cache is unencodable
    instead of embedded (no API call, no key read)."""

    def __init__(self, backend: TextEmbeddingBackend):
        self.id, self.dim = backend.id, backend.dim

    def embed_text(self, text: str) -> np.ndarray:
        raise LookupError(f"{text!r} is not in the text embedding cache (--offline)")


def make_offline(encoder) -> None:
    """Replace the API backend under every CachedTextBackend of the encoder (a text encoder's backend, a
    compositional encoder's name backend) with CacheOnlyTextBackend; cached texts are still served."""
    from cam.representations.text_backends import CachedTextBackend

    for attribute in ("backend", "name_backend"):
        backend = getattr(encoder, attribute, None)
        if isinstance(backend, CachedTextBackend):
            backend.backend = CacheOnlyTextBackend(backend.backend)


class Source:
    """name; levels(); points(level, operators) -> (points, {label: why it cannot be encoded})."""

    name: str
    trained_skills: frozenset[str] | None = None  # run sources: the run's skills

    def levels(self) -> list[Level]:
        raise NotImplementedError

    def level(self, name: str) -> Level:
        for level in self.levels():
            if level.name == name:
                return level
        raise KeyError(f"{self.name} has no level {name!r}; it has {[level.name for level in self.levels()]}")

    def point(self, level: Level, label: str, vector, origin: str, operators, text: str = "") -> EmbeddingPoint:
        trained = (None if self.trained_skills is None or origin in ("custom", "expression")
                   else any(name in self.trained_skills for name in operators))
        return EmbeddingPoint(label, level.kind, level.name, level.space, self.name,
                              np.asarray(vector, dtype=np.float32), origin, tuple(operators), trained, text)


def component_origin(entries: list[LabelledOperator], labels: list[str]) -> str:
    origins = {entry.origin for entry in entries if entry.label in labels}
    return "vocabulary" if "registered" in origins else ("custom" if "custom" in origins else "probe")


class FixedEncoderSource(Source):
    """An encoder with nothing learned (config["operator_encoder"]: one-hot, random, multi-hot or text). Levels:
        operator   the encoding, what the policy sees
        literal    multi-hot and text: each literal alone, as its one-literal operator (the containing operator's
                   name and parameters with only that precondition or effect), in the operators' space: multi-hot
                   is linear, so an operator is the sum of its literals (minus repeats of a name-only feature)
        feature    multi-hot: each feature an operator has (a basis vector of the encoding)
        predicate  text: each predicate's name, embedded as a text
    One-hot and random encode only the name: a custom operator keeping a registered name lands on that operator."""

    def __init__(self, name: str, config: dict, trained_skills=None):
        from cam.experiments.environment_setup import OPERATOR_ENCODERS

        self.name, self.config = name, config
        self.encoder_name = config["operator_encoder"]
        self.encoder = OPERATOR_ENCODERS[self.encoder_name](config, [])
        self.trained_skills = None if trained_skills is None else frozenset(trained_skills)

    def levels(self) -> list[Level]:
        dim = self.encoder.dim
        levels = [Level("operator", "operator", "operator", dim, f"the {self.encoder_name} encoding the policy sees")]
        if self.encoder_name in ("multi-hot", "text"):
            levels.append(Level("literal", "literal", "operator", dim, "each literal as its one-literal operator"))
        if self.encoder_name == "multi-hot":
            levels.append(Level("feature", "feature", "operator", dim, "each literal feature: a basis vector"))
        if self.encoder_name == "text":
            levels.append(Level("predicate", "predicate", "operator", dim, "each predicate's name as a text"))
        return levels

    def output(self, level: str, inputs: np.ndarray) -> np.ndarray:
        """What a level shows of the encoder's vectors (rows); TrainableRunSource applies W here."""
        return inputs

    def encode(self, operators: list[LabelledOperator]) -> tuple[list[LabelledOperator], np.ndarray, dict[str, str]]:
        encoded, vectors, unencodable = [], [], {}
        for entry in operators:
            try:
                vectors.append(self.encoder.encode(entry.operator))
                encoded.append(entry)
            except (ValueError, LookupError) as error:
                unencodable[entry.label] = str(error)
        return encoded, np.array(vectors, dtype=np.float64).reshape(len(vectors), self.encoder.dim), unencodable

    def points(self, level: str, operators: list[LabelledOperator]):
        spec = self.level(level)
        if spec.kind in ("feature", "predicate"):
            return self.vocabulary_points(spec, operators)
        if spec.kind == "literal":
            components = literal_components(operators).values()
            entries = [LabelledOperator(label, operator, component_origin(operators, members))
                       for label, operator, members in components]
            members = {label: labels for label, _, labels in components}
        else:
            entries, members = operators, {entry.label: [entry.label] for entry in operators}
        encoded, inputs, unencodable = self.encode(entries)
        points = [self.point(spec, entry.label, vector, entry.origin, members[entry.label], self.hover(entry))
                  for entry, vector in zip(encoded, self.output(level, inputs))]
        return points, unencodable

    def hover(self, entry: LabelledOperator) -> str:
        note = ""
        if entry.origin == "custom" and self.encoder_name in ("one-hot", "random"):
            note = f"\n{self.encoder_name} encodes the name only: the content is ignored"
        return to_pddl(entry.operator) + note

    def vocabulary_points(self, level: Level, operators: list[LabelledOperator]):
        """multi-hot's features and text's predicate names, each with the operators that have it."""
        members: dict = {}
        for entry in operators:
            keys = (operator_features(entry.operator) if level.kind == "feature"
                    else {predicate.name for _, predicate, _ in operator_literals(entry.operator)})
            for key in keys:
                members.setdefault(key, []).append(entry.label)
        keys, unencodable, rows = sorted(members, key=repr), {}, []
        for key in keys:
            if level.kind == "feature":
                row = np.zeros(self.encoder.dim)
                row[self.encoder.encoder.index[key]] = 1.0  # PaddedOperatorEncoder(PDDLMultiHotOperatorEncoder)
            else:
                try:
                    row = self.encoder.backend.embed_text(key)
                except LookupError as error:
                    unencodable[key] = str(error)
                    continue
            rows.append((feature_label(key) if level.kind == "feature" else key, key, row))
        inputs = np.array([row for _, _, row in rows], dtype=np.float64).reshape(len(rows), self.encoder.dim)
        points = [self.point(level, label, vector, component_origin(operators, members[key]), members[key])
                  for (label, key, _), vector in zip(rows, self.output(level.name, inputs))]
        return points, unencodable


def feature_label(feature: tuple) -> str:
    """e.g. "add on(0,1)", and "add on" for the name-only feature."""
    section, predicate = feature[0].replace("_", " "), feature[1]
    return f"{section} {predicate}({','.join(map(str, feature[2]))})" if len(feature) == 3 else f"{section} {predicate}"


class TrainableRunSource(FixedEncoderSource):
    """A --trainable-operator-embedding run: W (a features extractor's operator_embedding, initialised to the
    identity) applied to the encoder's vector x. Levels: operator (W x, what the policy sees), input (x), and for
    SAC operator-change ((W - I) x, what training added; PPO's orthogonal initialization replaces the identity, so
    W - I means nothing there), with the encoder's literal, feature and predicate levels through W, all in one
    space. With multi-hot, W x is the sum of the columns of W for the operator's features."""

    def __init__(self, name: str, config: dict, extractor, trained_skills=None):
        super().__init__(name, config, trained_skills)
        self.weight = extractor.operator_embedding.weight.detach().cpu().double().numpy()

    def levels(self) -> list[Level]:
        dim = self.encoder.dim
        levels = [Level(level.name, level.kind, level.space, dim, "W x: " + level.description) for level in super().levels()]
        levels.append(Level("input", "operator", "operator", dim, f"x: the {self.encoder_name} encoding before W"))
        if self.config["algorithm"] != "ppo":
            levels.append(Level("operator-change", "operator", "operator", dim, "(W - I) x: what training added"))
        return levels

    def output(self, level: str, inputs: np.ndarray) -> np.ndarray:
        if level == "input":
            return inputs
        if level == "operator-change":
            return inputs @ self.weight.T - inputs
        return inputs @ self.weight.T


# The levels with points per operator: level -> [(stage of the composition's intermediates, label suffix)]
OPERATOR_STAGES = {
    "and-sum": [("and-sum pre", " pre"), ("and-sum eff", " eff")],
    "and": [("and pre", " pre"), ("and eff", " eff")],
    "pre": [("pre", "")],
    "eff": [("eff", "")],
    "parameters": [("parameters", "")],
    "name": [("name", "")],
    "slots": [("slots", "")],
    "operator-raw": [("operator", "")],
}
KIND_SECTIONS = {3: ("pre", False), 4: ("pre", True), 5: ("eff", False), 6: ("eff", True)}  # structure.py's kinds


class CompositionalSource(Source):
    """A CompositionalPolicyFeaturesExtractor. Levels (c: --component-embedding-dim, D: --operator-embedding-dim,
    K: --slots), each one point per distinct component or per operator:
        type, variable, predicate   the component embeddings (one per type, parameter position, predicate)
        typed-variable              TYPED(variable, type), every parameter position with every type
        atom                        LITERAL(predicate, arguments): the predicate applied to its arguments, the same
                                    vector in any section and with either sign
        name                        the name's text embedding through its learned map (with the name included)
        tree: signed-literal (NOT applied to negated preconditions and delete effects), conjunct (phi of a signed
              literal), and-sum (each section's sum of phi: linear in its literals), and (rho of the sum), pre, eff,
              parameters
        slots: token (each component token plus its kind embedding: slot attention's inputs), slot, slots (the K
               slots flattened)
        geometric: slot, slots
        operator-raw                the composition's output, before normalization
        operator                    normalized: what the policy sees
    """

    def __init__(self, name: str, config: dict, extractor, trained_skills=None, all_atoms: bool = False):
        from cam.experiments.environment_setup import OPERATOR_ENCODERS

        self.name, self.config, self.extractor = name, config, extractor.eval()
        self.encoder = OPERATOR_ENCODERS["compositional"](config, [])
        self.layout = extractor.layout
        if self.encoder.layout != self.layout:
            raise ValueError(f"{name}: its extractor's layout differs from the one its config builds now "
                             "(a run saved by older code?)")
        self.composition = extractor.composition
        self.architecture = config["compositional_architecture"]
        self.trained_skills = None if trained_skills is None else frozenset(trained_skills)
        self.all_atoms = all_atoms

    def levels(self) -> list[Level]:
        c = self.composition.components.type_embedding.embedding_dim
        D, P = self.extractor.normalize.normalized_shape[0], self.layout.max_parameters
        levels = [
            Level("type", "type", "type", c, "each object type's embedding"),
            Level("variable", "variable", "variable", c, "each parameter position's embedding"),
            Level("typed-variable", "typed-variable", "typed-variable", c, f"TYPED(variable, type): {P} positions x types"),
            Level("predicate", "predicate", "predicate", c, "each predicate's embedding"),
            Level("atom", "atom", "literal", c, "LITERAL(predicate, arguments), in either section and sign"),
        ]
        if self.layout.name_dim:
            levels.append(Level("name", "name", "name", c, "the name's text embedding through its learned map"))
        if self.architecture == "tree":
            levels += [
                Level("signed-literal", "literal", "literal", c, "NOT(atom) for negated preconditions and deletes"),
                Level("conjunct", "literal", "conjunct-sum", c, "phi(signed literal), summed by AND"),
                Level("and-sum", "section", "conjunct-sum", c, "each section's sum of phi (linear in its literals)"),
                Level("and", "section", "and", c, "AND: rho of the sum"),
                Level("pre", "section", "pre", c, "PRE(AND(preconditions))"),
                Level("eff", "section", "eff", c, "EFF(AND(effects))"),
                Level("parameters", "section", "parameters", c, "PARAMETERS(typed variables)"),
            ]
        else:
            if self.architecture == "slots":
                levels.append(Level("token", "literal", "literal", c, "component token + kind: slot attention's inputs"))
            slots = self.composition.slot_attention.initial_slots.shape[0]
            levels += [Level("slot", "slot", "slot", c, "each slot after slot attention"),
                       Level("slots", "operator", "slots", slots * c, "the slots, flattened")]
        return levels + [
            Level("operator-raw", "operator", "operator-raw", D, "the composition's output, before normalization"),
            Level("operator", "operator", "operator", D, "normalized to magnitude 1: what the policy sees"),
        ]

    def points(self, level: str, operators: list[LabelledOperator]):
        import torch

        from cam.representations.compositional.structure import unpack

        spec = self.level(level)
        if spec.kind in ("type", "variable", "typed-variable", "predicate"):
            return self.vocabulary_points(spec, operators), {}
        entries, structures, unencodable = [], [], {}
        for entry in operators + (self.atom_probes() if level == "atom" and self.all_atoms else []):
            try:
                structures.append(self.encoder.encode(entry.operator))
                entries.append(entry)
            except (ValueError, LookupError) as error:
                unencodable[entry.label] = str(error)
        if not entries:
            return [], unencodable
        structure = torch.as_tensor(np.stack(structures))
        with torch.no_grad():
            if level == "operator":
                vectors = self.extractor.operator_embedding(structure).double().numpy()
                return [self.point(spec, e.label, v, e.origin, [e.label], to_pddl(e.operator))
                        for e, v in zip(entries, vectors)], unencodable
            parts = unpack(structure.float(), self.layout)
            stages = {} if level in ("atom", "name") else self.composition.intermediates(parts)
            components = self.composition.components
            if level == "atom":
                variables, _ = components.typed_variables(parts)
                stages["atom"] = components.literals(parts, variables)[0]
            if level == "name":
                stages["name"] = components.name(parts.name)
        stages = {key: value.double().numpy() for key, value in stages.items()}
        if level in OPERATOR_STAGES:
            points = [self.point(spec, entry.label + suffix, stages[stage][row], entry.origin, [entry.label],
                                 to_pddl(entry.operator))
                      for row, entry in enumerate(entries) for stage, suffix in OPERATOR_STAGES[level]]
            return points, unencodable
        if level == "slot":
            return [self.point(spec, f"{entry.label} slot {k + 1}", slot, entry.origin, [entry.label])
                    for entry, slots in zip(entries, stages["slot"]) for k, slot in enumerate(slots)], unencodable
        return self.component_points(spec, entries, structures, stages[level], operators), unencodable

    def component_points(self, level: Level, entries, structures, vectors, operators) -> list[EmbeddingPoint]:
        """Points of a per-component stage (B, N, c), once per distinct component. Literal rows are decoded from the
        structure arrays; the slots architecture's tokens are [name], parameters, then literal rows."""
        layout, found = self.layout, {}
        offset = (1 if layout.name_dim else 0) + layout.max_parameters if level.name == "token" else 0
        for entry, structure, rows in zip(entries, structures, vectors):
            operator = entry.operator
            literal_rows = structure[layout.max_parameters : layout.max_parameters + layout.max_literals * layout.literal_width]
            for row, values in enumerate(literal_rows.reshape(layout.max_literals, layout.literal_width).astype(int)):
                if values[0] == 0:
                    continue
                section, negated = KIND_SECTIONS[values[0]]
                arguments = tuple(operator.parameters[a - 1].name for a in values[2:] if a > 0)
                predicate = Predicate(layout.predicates[values[1] - 1], arguments)
                if level.name == "atom":
                    key, label = (values[1], tuple(values[2:])), str(predicate)
                elif level.name == "token":
                    key, label = tuple(values), literal_label(section, predicate, negated)
                else:
                    key, label = (negated, values[1], tuple(values[2:])), f"(not {predicate})" if negated else str(predicate)
                found.setdefault(key, (label, rows[offset + row], []))[2].append(entry.label)
            if level.name == "token":
                if layout.name_dim:
                    found.setdefault(("name", operator.name), (f"name {operator.name}", rows[0], []))[2].append(entry.label)
                for position, parameter in enumerate(operator.parameters):
                    key = ("parameter", position, parameter.type)
                    found.setdefault(key, (f"parameter {parameter.name} - {parameter.type}",
                                           rows[offset - layout.max_parameters + position], []))[2].append(entry.label)
        given = {entry.label for entry in operators}  # not --all-atoms' probes
        return [self.point(level, label, vector, component_origin(entries, members), [m for m in members if m in given])
                for label, vector, members in found.values()]

    def atom_probes(self) -> list[LabelledOperator]:
        """--all-atoms: a one-literal operator for every predicate applied to every ordering of distinct parameter
        positions, over max_parameters parameters ?x1 ... of the first type."""
        from cam.environments.fetch.fetch_predicate_evaluation_wrapper import FETCH_PREDICATE_ARITIES

        layout = self.layout
        parameters = " ".join(f"?x{i + 1} - {layout.types[0]}" for i in range(layout.max_parameters))
        probes = []
        for predicate in layout.predicates:  # the layout's predicates are the environment's (compositional_layout)
            for chosen in permutations(range(layout.max_parameters), FETCH_PREDICATE_ARITIES[predicate]):
                literal = f"({predicate}{''.join(f' ?x{i + 1}' for i in chosen)})"
                text = f"(:action atoms :parameters ({parameters}) :precondition (and {literal}) :effect (and))"
                probes.append(LabelledOperator(literal, parse_operator(text), "probe"))
        return probes

    def vocabulary_points(self, level: Level, operators: list[LabelledOperator]) -> list[EmbeddingPoint]:
        """The component embeddings of every predicate, type and parameter position (typed-variable: every position
        with every type), each with the operators that use it."""
        import torch

        components, layout = self.composition.components, self.layout
        if level.name == "predicate":
            keys = [(None, None, name) for name in layout.predicates]
        elif level.name == "type":
            keys = [(None, object_type, None) for object_type in layout.types]
        else:
            keys = [(position, object_type, None) for position in range(layout.max_parameters)
                    for object_type in (layout.types if level.name == "typed-variable" else [None])]
        with torch.no_grad():
            if level.name == "predicate":
                vectors = components.predicate_embedding.weight[1:]
            elif level.name == "type":
                vectors = components.type_embedding.weight[1:]
            elif level.name == "variable":
                vectors = components.variable_embedding.weight
            else:
                variables = components.variable_embedding(torch.tensor([key[0] for key in keys]))
                types = components.type_embedding(torch.tensor([layout.types.index(key[1]) + 1 for key in keys]))
                vectors = components.typed(torch.cat([variables, types], dim=-1))

        def uses(operator: PDDLOperator, position, object_type, predicate) -> bool:
            if predicate is not None:
                return predicate in {p.name for _, p, _ in operator_literals(operator)}
            return any((position is None or i == position) and (object_type is None or p.type == object_type)
                       for i, p in enumerate(operator.parameters))

        points = []
        for key, vector in zip(keys, vectors.detach().double().numpy()):
            position, object_type, predicate = key
            label = predicate or " - ".join(part for part in (
                None if position is None else f"parameter {position + 1}", object_type) if part)
            members = [entry.label for entry in operators if uses(entry.operator, *key)]
            points.append(self.point(level, label, vector, "vocabulary", members))
        return points


def parse_expression(expression: str, operators: dict[str, PDDLOperator]) -> list[tuple[float, str, PDDLOperator]]:
    """An expression such as "pickup - eff (not (raised ?o)) + eff (raised ?o)": operator labels and literals (pre or
    eff, then the literal) joined by + and - with spaces around them. Each literal becomes its one-literal operator on
    the first operator's name and parameters (a variable it does not have is added, of its first parameter's type).
    Returns (sign, term, operator) per term; ValueError for an unknown label or a malformed literal."""
    pieces = re.split(r"(?:^|\s+)([+-])\s+", expression.strip())
    signed = ([] if pieces[0] == "" else [("+", pieces[0])]) + list(zip(pieces[1::2], pieces[2::2]))
    if not signed or any(not term.strip() for _, term in signed):
        raise ValueError(f"an empty term in {expression!r}")
    base = next((operators[term.strip()] for _, term in signed if term.strip() in operators), None)
    result = []
    for sign, term in signed:
        term, weight = term.strip(), 1.0 if sign == "+" else -1.0
        if term in operators:
            result.append((weight, term, operators[term]))
            continue
        section, _, literal = term.partition(" ")
        if section not in ("pre", "eff") or base is None:
            raise ValueError(f"{term!r} is neither an operator label nor a literal (pre or eff, then the literal) in an "
                             "expression with an operator")
        names = {parameter.name for parameter in base.parameters}
        added = dict.fromkeys(token for token in tokenize(literal) if token.startswith("?") and token not in names)
        declared = " ".join(f"{p.name} - {p.type}" for p in base.parameters)
        declared += "".join(f" {variable} - {base.parameters[0].type}" for variable in added)
        sections = (f"(and {literal})", "(and)") if section == "pre" else ("(and)", f"(and {literal})")
        text = f"(:action {base.name} :parameters ({declared}) :precondition {sections[0]} :effect {sections[1]})"
        result.append((weight, term, parse_operator(text)))
    return result
