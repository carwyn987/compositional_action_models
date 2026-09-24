# Skill Environment — Specification

Scope: how a `Skill` becomes a Gymnasium RL task on top of an existing
Gymnasium environment (Fetch now, other simulators later), and how raw
observations become symbolic facts in the same vocabulary as the skill's
action model.

## 1. Principles

1. **Standard API only.** Every environment object is a `gymnasium.Env` or
   `gymnasium.Wrapper`. `reset(seed, options) -> (obs, info)` and
   `step(action) -> (obs, reward, terminated, truncated, info)`.
2. **No base-environment class.** Simulators are used as registered Gymnasium
   environments (`gym.make("FetchPickAndPlace-v4")`). Task semantics are added
   by one wrapper.
3. **Observations are not modified.** The wrapper never appends skill ids,
   embeddings, or operator features. Conditioning a policy on a skill is the
   `PolicyProvider`'s responsibility.
4. **One symbolic vocabulary.** Predicates reported by the environment use the
   exact names and arities used in the action-model files (`on-table`,
   `holding`, …) and are represented with the domain type `Predicate`.
5. **Success and preconditions come from the action model.** Per-skill code is
   limited to what the action model cannot express: setup, shaped reward, and
   an optional teacher.
6. **Dependency direction.** `domain` ← `skills` ← `environments`. `domain`
   imports neither Gymnasium nor NumPy.

## 2. Components and locations

```text
src/cam/
  domain/
    operators/grounding.py              # ground(), holds() — pure, no numpy
  skills/
    skill.py                            # Skill (extended, §5)
    setup/setup_procedure.py            # SetupProcedure, NoSetup, PolicySetup
    <name>_skill.py
  environments/
    predicates/predicate_evaluator.py   # PredicateEvaluator (abstract, §3)
    fetch/fetch_state.py                # FetchState + extract_fetch_state(obs)
    fetch/fetch_predicate_evaluator.py  # FetchPredicateEvaluator
    skill_environment.py                # SkillEnvironment, make_skill_environment
scripts/
  visualize_environment.py              # render a raw Gymnasium env + simple policy
```

## 3. PredicateEvaluator

### 3.1 Interface

```python
class PredicateEvaluator(ABC):
    objects: tuple[str, ...]                 # e.g. ("block0",)
    predicate_arities: dict[str, int]        # e.g. {"on-table": 1, "holding": 1, "clear": 1, "gripper-empty": 0}

    @abstractmethod
    def evaluate(self, observation) -> frozenset[Predicate]:
        """All ground predicates that are true in this observation (closed world)."""
```

### 3.2 Rules

- **Stateless.** `evaluate` is a pure function of the observation. No reference
  pose captured at reset. Facts that need history (e.g. "moved left") are not
  predicates of this evaluator.
- **Closed world.** Every ground atom over `objects` × `predicate_arities` not
  in the returned set is false.
- **PDDL semantics, not physical intuition.** The evaluator implements the
  meaning the action models assume. Example: `pickup` deletes `(clear ?o)`, so
  a held block is not `clear`, even though nothing is on top of it.
- **Owned by the environment type, not by a skill.** One evaluator per
  simulator/observation layout evaluates every predicate it supports; skills
  read facts, they never compute them.
- **Two stages for testability.** Observation layout knowledge lives only in
  `extract_fetch_state(obs) -> FetchState` (named fields: `gripper_position`,
  `finger_width`, `block_positions: dict[str, ndarray]`). Predicate logic
  consumes a `FetchState`, so it is unit-tested with hand-built states and no
  simulator.
- **Thresholds** live in one frozen dataclass per evaluator
  (`FetchPredicateThresholds`), not scattered constants.

### 3.3 Fetch predicates (single block, object name `block0`)

| Predicate | True when |
|---|---|
| `holding(b)` | gripper within `near_tolerance` of `b`, finger width below `open_width`, and `b` above table rest height + `lift_tolerance` |
| `on-table(b)` | `|b.z − table_rest_z| ≤ z_tolerance` and not `holding(b)` |
| `clear(b)` | no other block on top of `b` and not `holding(b)` |
| `gripper-empty` | no block `b` with `holding(b)` |

`table_rest_z` is a property of the Fetch table (block centre at rest ≈ 0.425),
not a value recorded at reset.

## 4. Grounding (domain)

An operator is written over variables (`?o`); the environment reports facts
over objects (`block0`). A **binding** maps each operator parameter to an
object: `{"?o": "block0"}`.

```python
def ground(op: PDDLOperator, binding: dict[str, str]) -> PDDLOperator
def preconditions_hold(op: PDDLOperator, facts: frozenset[Predicate]) -> bool   # op is ground
def effects_hold(op: PDDLOperator, facts: frozenset[Predicate]) -> bool         # op is ground
```

- `preconditions_hold`: every positive precondition ∈ facts, every negated one ∉ facts.
- `effects_hold`: every add effect ∈ facts, every delete effect ∉ facts.
- `ground` raises `ValueError` on an unbound or unknown variable.

These functions are format-specific; each symbolic action model format supplies
its own pair, selected through the loader's format table.

## 5. Skill (extended)

```python
class Skill(ABC):
    name: str                                   # abstract
    environment_id: str                         # abstract, e.g. "FetchPickAndPlace-v4"
    max_episode_steps: int = 100
    terminate_on_success: bool = True

    def __init__(self, config: dict)            # loads self.symbolic_action_model

    def setup_procedure(self) -> SetupProcedure           # default NoSetup()
    def choose_binding(self, objects, rng) -> dict[str, str]
    def preconditions_hold(self, facts, binding) -> bool   # from the action model
    def is_success(self, facts, binding) -> bool           # from the action model
    def reset_reward(self) -> None                         # clears reward state
    def reward(self, observation, action, facts, binding, success) -> tuple[float, dict[str, float]]
    def teacher(self) -> Policy | None                     # default None
```

- `reward` returns `(total, components)`; components are logged, total is used.
- All skills' rewards share one scale: per-step shaping in `[-1, 0]`, plus a
  success bonus of `+1`. Skills are compared under a shared policy, so unequal
  scales would bias training toward some skills.
- `is_success` defaults to the action model's `effects_hold`. A skill overrides
  it only when the action model under-specifies success; such an override is a
  signal that the action model should be revised.

## 6. SetupProcedure

Purpose: bring a freshly reset environment into a state where the skill's
preconditions hold.

```python
class SetupProcedure(ABC):
    @abstractmethod
    def run(self, env: gym.Env, observation, info) -> tuple[observation, int]:
        """Act on env; return the resulting observation and steps taken."""

class NoSetup(SetupProcedure)            # native reset already satisfies preconditions (pickup)
class PolicySetup(SetupProcedure)        # run a Policy for up to N steps (putdown: pickup teacher)
```

- Setup runs **inside** `SkillEnvironment.reset`, after the native reset.
- Setup steps are not episode steps. The `TimeLimit` wrapper sits outside
  `SkillEnvironment`, so steps taken during reset are never counted.
- Setup acts through the normal `step` API (physically valid states). State
  teleporting is not used: it produces grasps that the simulator does not hold.
- A teacher of one skill can be the setup of another (`putdown` is set up by
  running the `pickup` teacher).

## 7. SkillEnvironment

```python
class SkillEnvironment(gym.Wrapper):
    def __init__(self, env: gym.Env, skill: Skill, evaluator: PredicateEvaluator,
                 max_setup_attempts: int = 10)
```

### 7.1 `reset(seed=None, options=None)`

1. `obs, info = env.reset(seed, options)` (native reset).
2. `binding = skill.choose_binding(evaluator.objects, self.np_random)`.
3. `obs, setup_steps = skill.setup_procedure().run(env, obs, info)`.
4. `facts = evaluator.evaluate(obs)`; if not `skill.preconditions_hold(facts, binding)`,
   repeat from 1 (no seed on retries), up to `max_setup_attempts`, then raise
   `RuntimeError`.
5. `skill.reset_reward()`.
6. Return `obs` and `info` augmented per §7.3.

### 7.2 `step(action)`

1. `obs, _, terminated, truncated, info = env.step(action)` (native reward discarded).
2. `facts = evaluator.evaluate(obs)`.
3. `success = skill.is_success(facts, binding)`.
4. `reward, components = skill.reward(obs, action, facts, binding, success)`.
5. `terminated = terminated or (success and skill.terminate_on_success)`.
6. Return with `info` augmented per §7.3.

### 7.3 `info` contract

| Key | Type | Present |
|---|---|---|
| `skill` | `str` | reset, step |
| `binding` | `dict[str, str]` | reset, step |
| `facts` | `frozenset[Predicate]` | reset, step |
| `is_success` | `float` (0/1; Stable-Baselines3 convention) | reset, step |
| `reward_components` | `dict[str, float]` | step |
| `setup_steps`, `setup_attempts` | `int` | reset |

### 7.4 Construction

```python
def make_skill_environment(skill: Skill, render_mode: str | None = None) -> gym.Env:
    base = gym.make(skill.environment_id, render_mode=render_mode)   # its TimeLimit removed
    env = SkillEnvironment(base, skill, evaluator_for(skill.environment_id))
    return gym.wrappers.TimeLimit(env, skill.max_episode_steps)
```

`evaluator_for` maps environment ids to evaluators
(`FetchPickAndPlace-v4 → FetchPredicateEvaluator(objects=("block0",))`).
`gym.make` always applies the registered `TimeLimit` (50 steps for Fetch), so
the factory unwraps that layer before wrapping.

### 7.5 Multi-skill

One `SkillEnvironment` per skill. Skill selection across episodes belongs to
the `SkillSampler`, outside the environment.

## 8. Tests

| Target | Test | Marker |
|---|---|---|
| grounding | ground/unbound variable; preconditions/effects hold incl. negation and delete | unit |
| `FetchPredicateEvaluator` | each predicate at both sides of its thresholds, from hand-built `FetchState` | unit |
| `extract_fetch_state` | field values from a synthetic 25-dim observation | unit |
| vocabulary | every predicate in every skill's action model is in the evaluator's `predicate_arities` with the same arity | unit |
| `SkillEnvironment` | on a small fake `gym.Env`: setup runs; preconditions verified; retries then raises; `info` keys; success terminates; setup steps not counted by `TimeLimit` | unit |
| Fetch reset | pickup preconditions hold after native reset for all of 20 seeds | integration |
| Fetch setup | putdown preconditions hold after `PolicySetup(pickup teacher)` | integration |
| Fetch rollout | pickup teacher reaches `is_success` within `max_episode_steps` | integration |

`integration` tests use MuJoCo, take seconds, and are selectable with
`-m integration` / excludable with `-m "not integration"`.

## 9. Open decisions

1. **Push predicates.** `(moved-left ?o)` is an event, not a state; it
   violates §3.2 (stateless). Either model a spatial state relation
   (`(left-of ?o ?ref)` with a reference object/region in the scene) or allow a
   per-episode reference recorded by the wrapper.
2. **Goal keys.** Fetch observations include `desired_goal` (a random target
   irrelevant to these skills). Keep as-is, or drop with a
   `FilterObservation` wrapper.
3. **Binding choice with several objects.** Random over valid bindings, or
   fixed by the setup procedure (e.g. the block it placed).
