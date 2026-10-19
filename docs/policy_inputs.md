# Policy inputs

What the policy network receives each step, and why. The observation is built
by `cam.training.policy_observation_wrapper.PolicyObservationWrapper` as a
dictionary of three fixed-size vectors:

| Key | Contents | Changes when | Built by |
|---|---|---|---|
| `observation` | the environment's state vector (Fetch: gripper, fingers, every block) | every step | the environment (Fetch goal keys are dropped by `FetchEnvStateAnnotationWrapper`) |
| `operator_embedding` | the **lifted** action model, encoded by `--operator-encoder` | the skill changes | an `OperatorEncoder` (`cam.representations`) |
| `grounding` | which object fills each of the action model's parameters | the episode's grounding changes | `GroundingEncoder` (`cam.representations`) |

The operator embedding says *what kind* of action this is (pickup vs putdown);
the grounding says *which objects* it applies to (block2 vs block0). The two are
linked by **parameter position**: parameter 0 in the operator embedding is
grounding slot 0, parameter 1 is slot 1, and so on (worked example in
[How the operator embedding and the grounding line up](#how-the-operator-embedding-and-the-grounding-line-up)).

## Operator embedding

Every encoder (one-hot, random, text, multi-hot, trainable, compositional) is
described in `docs/operator_embeddings.md`, with worked examples. They share one
size, `--operator-embedding-dim` (default 256), except the compositional
condition, whose observation entry is a structure array that the policy composes
into a 256-value embedding. `PolicyObservationWrapper` encodes each distinct
lifted operator it sees, so a repaired operator (a new, different object) gets
its own encoding.

The default `--operator-encoder`, multi-hot (`PDDLMultiHotOperatorEncoder`),
gives each operator a vector over a fixed vocabulary of literal features, with
1 where it has the feature and 0 where it does not. A feature is a literal of
one section (`precondition`, `negated_precondition`, `add`, `delete`),
described two ways:

- **positional**: `(section, predicate, parameter positions)`, e.g.
  `("add", "on", (0, 1))` = "the object in parameter 0 ends up on the object in
  parameter 1". Variable names (`?o`, `?b`) are irrelevant; positions are canonical.
- **name only**: `(section, predicate)`, e.g. `("add", "on")`. Shared by more
  operators, so a held-out operator still activates features seen in training.

Example, the stack operator (used again below):

```lisp
(:action stack
    :parameters (?o - block ?u - block)
    :precondition (and (holding ?o) (clear ?u))
    :effect (and (on ?o ?u) (clear ?o) (gripper-empty) (not (holding ?o)) (not (clear ?u))))
```

For Fetch with `--max-operator-arity 3` the vocabulary has 192 features. It is
sorted (section, predicate, then positions, the name-only feature first), so
each section is one block of 48 features in the same order: feature i is at
index i in add, 48 + i in delete, 96 + i in negated_precondition and 144 + i in
precondition. Every feature, with stack's value (1 in bold):

| i | Feature | add (i) | delete (48 + i) | negated_precondition (96 + i) | precondition (144 + i) |
|---|---|---|---|---|---|
| 0 | `at-back` (name only) | 0 | 0 | 0 | 0 |
| 1 | `at-back`(0) | 0 | 0 | 0 | 0 |
| 2 | `at-back`(1) | 0 | 0 | 0 | 0 |
| 3 | `at-back`(2) | 0 | 0 | 0 | 0 |
| 4 | `at-front` (name only) | 0 | 0 | 0 | 0 |
| 5 | `at-front`(0) | 0 | 0 | 0 | 0 |
| 6 | `at-front`(1) | 0 | 0 | 0 | 0 |
| 7 | `at-front`(2) | 0 | 0 | 0 | 0 |
| 8 | `at-left` (name only) | 0 | 0 | 0 | 0 |
| 9 | `at-left`(0) | 0 | 0 | 0 | 0 |
| 10 | `at-left`(1) | 0 | 0 | 0 | 0 |
| 11 | `at-left`(2) | 0 | 0 | 0 | 0 |
| 12 | `at-right` (name only) | 0 | 0 | 0 | 0 |
| 13 | `at-right`(0) | 0 | 0 | 0 | 0 |
| 14 | `at-right`(1) | 0 | 0 | 0 | 0 |
| 15 | `at-right`(2) | 0 | 0 | 0 | 0 |
| 16 | `beside` (name only) | 0 | 0 | 0 | 0 |
| 17 | `beside`(0, 1) | 0 | 0 | 0 | 0 |
| 18 | `beside`(0, 2) | 0 | 0 | 0 | 0 |
| 19 | `beside`(1, 0) | 0 | 0 | 0 | 0 |
| 20 | `beside`(1, 2) | 0 | 0 | 0 | 0 |
| 21 | `beside`(2, 0) | 0 | 0 | 0 | 0 |
| 22 | `beside`(2, 1) | 0 | 0 | 0 | 0 |
| 23 | `clear` (name only) | **1** | **1** | 0 | **1** |
| 24 | `clear`(0) | **1** | 0 | 0 | 0 |
| 25 | `clear`(1) | 0 | **1** | 0 | **1** |
| 26 | `clear`(2) | 0 | 0 | 0 | 0 |
| 27 | `gripper-empty` (name only) | **1** | 0 | 0 | 0 |
| 28 | `gripper-empty`() | **1** | 0 | 0 | 0 |
| 29 | `holding` (name only) | 0 | **1** | 0 | **1** |
| 30 | `holding`(0) | 0 | **1** | 0 | **1** |
| 31 | `holding`(1) | 0 | 0 | 0 | 0 |
| 32 | `holding`(2) | 0 | 0 | 0 | 0 |
| 33 | `on` (name only) | **1** | 0 | 0 | 0 |
| 34 | `on`(0, 1) | **1** | 0 | 0 | 0 |
| 35 | `on`(0, 2) | 0 | 0 | 0 | 0 |
| 36 | `on`(1, 0) | 0 | 0 | 0 | 0 |
| 37 | `on`(1, 2) | 0 | 0 | 0 | 0 |
| 38 | `on`(2, 0) | 0 | 0 | 0 | 0 |
| 39 | `on`(2, 1) | 0 | 0 | 0 | 0 |
| 40 | `on-table` (name only) | 0 | 0 | 0 | 0 |
| 41 | `on-table`(0) | 0 | 0 | 0 | 0 |
| 42 | `on-table`(1) | 0 | 0 | 0 | 0 |
| 43 | `on-table`(2) | 0 | 0 | 0 | 0 |
| 44 | `raised` (name only) | 0 | 0 | 0 | 0 |
| 45 | `raised`(0) | 0 | 0 | 0 | 0 |
| 46 | `raised`(1) | 0 | 0 | 0 | 0 |
| 47 | `raised`(2) | 0 | 0 | 0 | 0 |

Each of stack's 7 literals sets its positional feature and its section's
name-only feature for that predicate: 14 of 192. `PaddedOperatorEncoder` then
appends 64 zeros, up to `--operator-embedding-dim` (256). The float32 vector in
`obs["operator_embedding"]`, 16 values per row:

```text
index      values                              part
  0-15     0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0    add
 16-31     0 0 0 0 0 0 0 1  1 0 0 1 1 0 0 0    add
 32-47     0 1 1 0 0 0 0 0  0 0 0 0 0 0 0 0    add
 48-63     0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0    delete
 64-79     0 0 0 0 0 0 0 1  0 1 0 0 0 1 1 0    delete
 80-95     0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0    delete
 96-111    0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0    negated_precondition
112-127    0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0    negated_precondition
128-143    0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0    negated_precondition
144-159    0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0    precondition
160-175    0 0 0 0 0 0 0 1  0 1 0 0 0 1 1 0    precondition
176-191    0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0    precondition
192-207    0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0    padding (PaddedOperatorEncoder)
208-223    0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0    padding (PaddedOperatorEncoder)
224-239    0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0    padding (PaddedOperatorEncoder)
240-255    0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0    padding (PaddedOperatorEncoder)
```

that is, 1 at indices 23, 24, 27, 28, 33, 34, 71, 73, 77, 78, 167, 169, 173, 174 and 0 everywhere else.

How the vocabulary is built, its size for other maximum arities and the
operators it rejects are in `docs/operator_embeddings.md`.

## Grounding (dereferenced parameter slots)

The grounding maps each parameter, in order, to the object bound to it, e.g.
`pickup(block2)` maps parameter 0 → block2. Rather than passing only a pointer
(the object's index) and leaving the network to look the object up in the
observation, each slot carries the pointed-to object's own features:

```text
slot = [ used (1) | type one-hot (n_types) | object index one-hot (max_objects) | object features (feature_dim) ]
grounding = slot_0 + slot_1 + ... + slot_{max_arity-1}   (concatenated)
```

- **used**: 1 for a parameter the operator has, 0 for padding. There are
  `max_arity` slots (`--max-operator-arity`, default 3, shared with the operator
  embedding); operators with fewer parameters get zero-filled slots, so every
  skill's input has the same size.
- **type one-hot**: the parameter's type (`block`, later e.g. `tray`). Each type
  shares the slot; feature vectors are zero-padded to the widest type.
- **object index one-hot**: which object it is (its position in
  `info["objects"]`), relating the slot to that object's part of the observation.
  Its length is the environment's maximum object count (Fetch: 5 blocks), so it
  does not change with the number of blocks in a run.
- **object features**: the object's current features from
  `info["object_features"]` (Fetch blocks: position and position relative to the
  gripper). This is the object's *current* state, not a target.

Example with three blocks, `pickup(block2)`, `max_arity = 1`, `max_objects = 3`:

```text
[ 1 | 1 | 0, 0, 1 | x, y, z, dx, dy, dz ]
```

Only parameters get slots. The gripper is not a parameter of any current
operator; its state is in `observation`. An operator with an arm parameter
would give the arm a slot like any other object.

## How the operator embedding and the grounding line up

Neither vector contains the other. The operator embedding is computed from the
lifted operator alone, once per operator, so `stack(block2, block0)` and
`stack(block1, block2)` get the same operator embedding; the grounding holds
objects and no predicates. The two share one convention, the order of the
operator's `:parameters` list, and each encoder follows it on its own:

- the operator encoder writes each variable as its position in that list
  (`?o` → 0, `?u` → 1);
- the grounded operator lists its objects in the same order
  (`GroundedSymbolicActionModel.arguments`), and the grounding encoder writes
  the object bound to parameter i into slot i.

Example: `stack(block2, block0)` (stack is shown under
[Operator embedding](#operator-embedding)) with the multi-hot encoder, three
blocks, `max_arity = 3`, `max_objects = 3`.

| Position | Variable | Positional features that mention it | Grounding slot |
|---|---|---|---|
| 0 | `?o` | precondition `holding`(0), add `on`(0, 1), add `clear`(0), delete `holding`(0) | slot 0: block2 |
| 1 | `?u` | precondition `clear`(1), add `on`(0, 1), delete `clear`(1) | slot 1: block0 |
| 2 | none | none | slot 2: empty |

Add `gripper-empty`() mentions no parameter. The operator embedding sets these
7 positional features and their 7 name-only features (14 of 192), and the
grounding is

```text
          used   type one-hot   object index one-hot       object features
                 (block)        (block0, block1, block2)   (position, position relative to the gripper)
slot 0  [ 1    | 1            | 0, 0, 1                  | block2's position, block2's position relative to the gripper ]
slot 1  [ 1    | 1            | 1, 0, 0                  | block0's position, block0's position relative to the gripper ]
slot 2  [ 0    | 0            | 0, 0, 0                  | 0, 0, 0, 0, 0, 0 ]
```

Read together: hold the block in slot 0, and end with it on the block in slot 1.

Nothing joins the two explicitly. The policy's input is the concatenation
`[ grounding | observation | operator embedding ]` (observation keys in sorted
order, for every features extractor), and its fully connected network has to
learn the link from reward: when add `on`(0, 1) is set, the target is slot 1's
position. Where an entry sits in that input does not matter to such a network;
reordering the inputs only reorders the first layer's weights.

- Roles keep their slot across operators that list parameters in the same
  order: in `stack` and `place-beside`, parameter 0 is the block being placed
  and parameter 1 the block it is placed relative to.
- Reordering an operator's parameters (`:parameters (?u - block ?o - block)`)
  changes both vectors consistently (add `on`(1, 0); block2 moves to slot 1):
  the same action, but a different input.
- Only the multi-hot and compositional encoders refer to parameter positions.
  One-hot and random describe no parameters, so the network learns each
  operator's slot roles separately; the text encoder sees variable names (`?o`),
  related to slots only by their order in the text.

## How the input grows

With P predicates of arity at most K, maximum operator arity n, T types, M
objects and F object features:

| Part | Size | Fetch (P = 11, K = 2, T = 1, M = 5, F = 6) |
|---|---|---|
| multi-hot operator embedding | 4 × sum over predicates of (1 + n! / (n - k)!), k the predicate's arity; ≤ 4P(1 + n^K) | 8n² + 24n + 48: 80, 128, 192, 272, 368 for n = 1 to 5 |
| grounding | n(1 + T + M + F) | 13n: 39 for n = 3 |
| observation | linear in the number of objects | gripper, fingers, every block |

- Nothing grows with the number of operators or skills, and nothing is
  exponential in what changes in practice. The multi-hot is one entry per
  (section, predicate, parameter tuple), so it is polynomial in n of degree K,
  and exponential only in predicate arity, which stays small. At n = 5 it
  exceeds the default `--operator-embedding-dim` of 256.
- It is readable (every entry has a name) but not compressed: stack sets 14 of
  192 entries, the name-only features repeat the positional ones (kept so a
  held-out operator shares features with training operators), and most entries
  are never set. An entry that is always 0 costs a fully connected network only
  its weights.
- The compositional encoder (`docs/operator_embeddings.md`) does not grow
  combinatorially: its structure array holds each literal as ids, linear in its
  maxima, and the learned embedding has the fixed size
  `--operator-embedding-dim`. A vector-symbolic encoding (e.g. holographic
  reduced representations, Plate 1995) would too, without learning: fixed
  random vectors per section, predicate, argument and parameter position, bound
  by elementwise multiplication and summed over literals. Its size is fixed for
  any n, K or P; more literals add interference, not entries. Not implemented.
- None of these makes the link to the grounding explicit. A per-parameter
  encoding would: one token per parameter, its operator features (section,
  predicate, argument index: 4 × sum of predicate arities, independent of n)
  joined with its grounding slot, read by a network shared across parameters
  with attention between them. Not implemented.

## Requirements on the environment

`PolicyObservationWrapper` reads only `info` keys, so it works with any
environment whose wrappers provide them:

| Key | Provided by (Fetch) |
|---|---|
| `objects` (name → type) | `FetchEnvStateAnnotationWrapper` |
| `object_features` (name → vector) | `FetchEnvStateAnnotationWrapper` |
| `grounded_action_model` | `SkillEnvironment` |
