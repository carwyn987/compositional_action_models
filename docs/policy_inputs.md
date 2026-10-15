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

Example: `stack(block2, block0)` with the multi-hot encoder (its positional and
name-only features are in `docs/operator_embeddings.md`), three blocks,
`max_arity = 3`, `max_objects = 3`.

```lisp
(:action stack
    :parameters (?o - block ?u - block)
    :precondition (and (holding ?o) (clear ?u))
    :effect (and (on ?o ?u) (clear ?o) (gripper-empty) (not (holding ?o)) (not (clear ?u))))
```

| Position | Variable | Positional features that mention it | Grounding slot |
|---|---|---|---|
| 0 | `?o` | precondition `holding`(0), add `on`(0, 1), add `clear`(0), delete `holding`(0) | slot 0: block2 |
| 1 | `?u` | precondition `clear`(1), add `on`(0, 1), delete `clear`(1) | slot 1: block0 |
| 2 | none | none | slot 2: empty |

Add `gripper-empty`() mentions no parameter. The operator embedding sets these
7 positional features and their 7 name-only features (14 of 192), and the
grounding is

```text
slot 0  [ 1 | 1 | 0, 0, 1 | block2's position, block2's position relative to the gripper ]
slot 1  [ 1 | 1 | 1, 0, 0 | block0's position, block0's position relative to the gripper ]
slot 2  [ 0 | 0 | 0, 0, 0 | 0, 0, 0, 0, 0, 0 ]
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
