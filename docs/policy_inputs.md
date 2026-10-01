# Policy inputs

What the policy network receives each step, and why. The observation is built
by `cam.training.policy_observation_wrapper.PolicyObservationWrapper` as a
dictionary of three fixed-size vectors:

| Key | Contents | Changes when | Built by |
|---|---|---|---|
| `observation` | the environment's state vector (Fetch: gripper, fingers, every block) | every step | the environment (Fetch goal keys are dropped by `FetchEnvStateAnnotationWrapper`) |
| `operator_embedding` | the **lifted** action model's structure | the skill changes | an `OperatorEncoder` (`cam.representations`) |
| `grounding` | which object fills each of the action model's parameters | the episode's grounding changes | `GroundingEncoder` (`cam.representations`) |

The operator embedding says *what kind* of action this is (pickup vs putdown);
the grounding says *which objects* it applies to (block2 vs block0). The two are
linked by **parameter position**: parameter 0 in the operator embedding is
grounding slot 0, parameter 1 is slot 1, and so on.

## Operator embedding (positional multi-hot)

`PDDLMultiHotOperatorEncoder` lists every feature appearing in the training
operators and gives each operator a vector with 1 where it has the feature and
0 where it does not. A feature is a literal of one section, described two ways:

- **positional**: `(section, predicate, parameter positions)`, e.g.
  `("add", "on", (0, 1))` = "the object in parameter 0 ends up on the object in
  parameter 1". Variable names (`?o`, `?b`) are irrelevant; positions are canonical.
- **name only**: `(section, predicate)`, e.g. `("add", "on")`. Shared by more
  operators, so a held-out operator still activates features seen in training.

Sections: `precondition`, `negated_precondition`, `add`, `delete`.

Example over pickup and putdown, showing only the features they set
(positional features shown; each also sets the matching name-only feature):

| Feature | pickup `(?o)` | putdown `(?o)` |
|---|---|---|
| precondition `on-table`(0) | 1 | 0 |
| precondition `clear`(0) | 1 | 0 |
| precondition `gripper-empty`() | 1 | 0 |
| precondition `holding`(0) | 0 | 1 |
| add `holding`(0) | 1 | 0 |
| add `on-table`(0) | 0 | 1 |
| add `clear`(0) | 0 | 1 |
| add `gripper-empty`() | 0 | 1 |
| delete `on-table`(0) | 1 | 0 |
| delete `clear`(0) | 1 | 0 |
| delete `gripper-empty`() | 1 | 0 |
| delete `holding`(0) | 0 | 1 |

The vocabulary does **not** depend on the training operators. It is every
feature any operator could have, given the predicates the environment can
evaluate (Fetch: `FETCH_PREDICATE_ARITIES`) and a maximum operator arity
(`--max-operator-arity`, default 3):

    size = 4 sections x sum over predicates of (1 + max_arity! / (max_arity - arity)!)

For Fetch that is 36 / 56 / 84 features for a maximum arity of 1 / 2 / 3. So
training on n operators and testing on an (n+1)-th uses the same indices, and
an operator repaired by adding or removing preconditions or effects is simply
re-encoded: only the features of the changed literals move.
`PolicyObservationWrapper` encodes each distinct lifted operator it sees, so a
repaired operator (a new, different object) gets its own encoding.

Operators the vocabulary cannot represent are rejected with `ValueError`
rather than partially encoded: more parameters than the maximum arity, a
predicate the environment cannot evaluate, a constant argument, or a repeated
parameter such as `(on ?x ?x)`.

`OneHotOperatorEncoder` is the baseline: one entry per training skill, no
structure. An operator it was not built with (e.g. a held-out one) raises
`ValueError`. Operators are identified by name, so a repaired operator that keeps
its name gets the same encoding as before.

## Grounding (dereferenced parameter slots)

The grounding maps each parameter, in order, to the object bound to it, e.g.
`pickup(block2)` maps parameter 0 → block2. Rather than passing only a pointer
(the object's index) and leaving the network to look the object up in the
observation, each slot carries the pointed-to object's own features:

```text
slot = [ used (1) | type one-hot (n_types) | object index one-hot (max_objects) | object features (feature_dim) ]
grounding = slot_0 + slot_1 + ... + slot_{max_arity-1}
```

- **used**: 1 for a parameter the operator has, 0 for padding. There are
  `max_arity` slots (`--max-operator-arity`, shared with the operator
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

Example with three blocks, `pickup(block2)`, `max_arity = 1`:

```text
[ 1 | 1 | 0, 0, 1 | x, y, z, dx, dy, dz ]
```

Only parameters get slots. The gripper is not a parameter of any current
operator; its state is in `observation`. An operator with an arm parameter
would give the arm a slot like any other object.

## Requirements on the environment

`PolicyObservationWrapper` reads only `info` keys, so it works with any
environment whose wrappers provide them:

| Key | Provided by (Fetch) |
|---|---|
| `objects` (name → type) | `FetchEnvStateAnnotationWrapper` |
| `object_features` (name → vector) | `FetchEnvStateAnnotationWrapper` |
| `grounded_action_model` | `SkillEnvironment` |
