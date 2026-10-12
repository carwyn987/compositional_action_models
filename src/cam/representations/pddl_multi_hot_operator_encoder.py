"""Multi-hot encoder for PDDL operators, referring to parameters by position.

An embedding method with no learned parts: an operator becomes the set of its
literal features, and the vector has a 1 for each feature it has (a "bag of
literals"; nothing is weighted or combined). Learned methods that embed each
literal and aggregate them (mean, attention, slots, ...) are other
OperatorEncoder implementations with the same interface.

The vocabulary is every feature that any operator over the given predicates
could have, so it does not depend on which operators are trained on: an
operator added later, or one repaired by adding or removing preconditions and
effects, is encoded in the same feature space without rebuilding the encoder.

Pipeline, with the putdown operator from the action model library:

    operator (PDDL)      (:action putdown :parameters (?o - block)
                            :precondition (and (holding ?o))
                            :effect (and (on-table ?o) (clear ?o) (gripper-empty) (not (holding ?o))))

    operator_features()  {("precondition", "holding", (0,)), ("precondition", "holding"),
                          ("add", "on-table", (0,)),        ("add", "on-table"),
                          ("add", "clear", (0,)),           ("add", "clear"),
                          ("add", "gripper-empty", ()),     ("add", "gripper-empty"),
                          ("delete", "holding", (0,)),      ("delete", "holding")}

    encode()             0/1 vector over the vocabulary with these 10 features set
                         (see PDDLMultiHotOperatorEncoder)

Each literal gives two features:
    positional  (section, predicate, parameter positions)   e.g. ("add", "on", (0, 1))
    name only   (section, predicate)                        e.g. ("add", "on")
Sections: precondition, negated_precondition, add, delete. Parameter positions
replace variable names (?o is parameter 0). See docs/policy_inputs.md.
"""

from itertools import permutations

import numpy as np

from cam.domain.pddl.pddl import PDDLOperator
from cam.domain.symbols import Predicate
from cam.representations.operator_encoder import OperatorEncoder

SECTIONS = ("precondition", "negated_precondition", "add", "delete")

Feature = tuple


def operator_features(operator: PDDLOperator) -> set[Feature]:
    """The positional and name-only features of an operator's literals.

    Example, stack (?o - block ?b - block) with effect (on ?o ?b):
        ("add", "on", (0, 1))   parameter 0 ends up on parameter 1
        ("add", "on")           something ends up on something
    A literal argument that is not a parameter (a constant) is kept as its name.
    """
    positions = {param.name: i for i, param in enumerate(operator.parameters)}

    def literal_features(section: str, predicate: Predicate) -> set[Feature]:
        arguments = tuple(positions.get(arg, arg) for arg in predicate.args)
        return {(section, predicate.name, arguments), (section, predicate.name)}

    features: set[Feature] = set()
    for precondition in operator.preconditions:
        section = "negated_precondition" if precondition.negated else "precondition"
        features |= literal_features(section, precondition.predicate)
    for effect in operator.effects:
        features |= literal_features("delete" if effect.delete else "add", effect.predicate)
    return features


def vocabulary(predicate_arities: dict[str, int], max_arity: int) -> list[Feature]:
    """Every feature of every section and predicate, with positions over up to
    max_arity distinct parameters, in a fixed (sorted) order."""
    features = set()
    for section in SECTIONS:
        for predicate, arity in predicate_arities.items():
            features.add((section, predicate))
            features |= {(section, predicate, positions) for positions in permutations(range(max_arity), arity)}
    return sorted(features, key=repr)


class PDDLMultiHotOperatorEncoder(OperatorEncoder):
    """Vocabulary from the predicates an environment can evaluate and a maximum
    operator arity; encode() puts a 1 at each of an operator's features.

    Size: 4 sections x (1 name-only + max_arity!/(max_arity - k)! positional
    features per predicate of arity k). For Fetch (holding/1, on-table/1,
    clear/1, on/2, gripper-empty/0, raised/1, beside/2): max_arity 1 -> 48,
    2 -> 80, 3 -> 128.

    Example, Fetch predicates with max_arity=1 (48 features); the features set
    for pickup and putdown (each positional / name-only pair is one row, every
    other feature is 0):

        feature                              pickup  putdown
        precondition on-table(0)               1       0
        precondition clear(0)                  1       0
        precondition gripper-empty()           1       0
        precondition holding(0)                0       1
        add holding(0)                         1       0
        add on-table(0)                        0       1
        add clear(0)                           0       1
        add gripper-empty()                    0       1
        delete on-table(0)                     1       0
        delete clear(0)                        1       0
        delete gripper-empty()                 1       0
        delete holding(0)                      0       1

    Adding the precondition (not (on ?o ?x)) to an operator would set the
    negated_precondition on(0, 1) and on features; nothing else moves.

    encode() raises ValueError for an operator with more than max_arity
    parameters or a literal outside the vocabulary (a predicate the environment
    cannot evaluate, a constant argument, or a repeated parameter).
    """

    def __init__(self, predicate_arities: dict[str, int], max_arity: int):
        self.max_arity = max_arity
        self.features: list[Feature] = vocabulary(predicate_arities, max_arity)
        self.index = {feature: i for i, feature in enumerate(self.features)}

    @property
    def dim(self) -> int:
        return len(self.features)

    def encode(self, operator: PDDLOperator) -> np.ndarray:
        if len(operator.parameters) > self.max_arity:
            raise ValueError(f"{operator.name} has {len(operator.parameters)} parameters; max_arity={self.max_arity}")
        features = operator_features(operator)
        unknown = sorted((f for f in features if f not in self.index), key=repr)
        if unknown:
            raise ValueError(f"{operator.name} has features outside the vocabulary: {unknown}")
        encoding = np.zeros(self.dim, dtype=np.float32)
        encoding[[self.index[feature] for feature in features]] = 1.0
        return encoding
