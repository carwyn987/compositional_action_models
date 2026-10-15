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
grounding slot 0, parameter 1 is slot 1, and so on.

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

## Requirements on the environment

`PolicyObservationWrapper` reads only `info` keys, so it works with any
environment whose wrappers provide them:

| Key | Provided by (Fetch) |
|---|---|
| `objects` (name → type) | `FetchEnvStateAnnotationWrapper` |
| `object_features` (name → vector) | `FetchEnvStateAnnotationWrapper` |
| `grounded_action_model` | `SkillEnvironment` |
