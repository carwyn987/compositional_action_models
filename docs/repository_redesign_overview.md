# Compositional Action Models — Repository Redesign Overview

## Purpose
Build a modular research codebase for learning robotic control policies from symbolic action models. The core research question is whether a structured action-model representation (e.g. PDDL operator) can be embedded and used to:

1. condition a shared policy,
2. modulate a shared policy (e.g. FiLM), or
3. generate policy parameters directly via a meta-network / hypernetwork.

The codebase must support zero-shot evaluation on unseen operators, fine-tuning speed, multi-skill training, policy-similarity analysis, and controlled ablations across representation / policy / training choices.

## Core Design Principles
- Single-responsibility modules; avoid large manager classes.
- Define small interfaces first; concrete implementations are swappable.
- Domain semantics must not depend on PyTorch, Gym, RL algorithms, or logging.
- Evaluation is separate from training.
- Experiment configuration is serialized (YAML/JSON), reproducible, and is the only layer that assembles the whole system.
- Prefer vertical incremental development: get a tiny end-to-end path working early, then add complexity.

## Core Concepts / Interfaces

### Domain
- `PDDLOperator`: name, typed parameters, preconditions, add effects, delete effects.
- `Predicate`, `TypedParameter`, `Effect`.
- `SkillSpec`: task definition containing name, operator, environment spec, setup procedure, reward, success condition, optional teacher.
- Keep room for future frontends (PDDL 2.1/3.1/PDDL+) by having a generic `TaskLevelOperator` or language frontend abstraction.

### Representation
`OperatorEncoder` maps a task-level operator to an `OperatorRepresentation`.

Implementations:
- whole-string encoder,
- compositional small-vector encoder,
- slot-based structured encoder,
- later: graph-based encoder.

Compositional encoders should support interchangeable aggregation strategies such as mean pooling, DeepSets, attention, slots, and later graph/message-passing methods.

### Policy
`Policy` is intentionally minimal: observation -> action.

`PolicyProvider` maps operator representation / skill information to an executable policy.

Implementations:
- shared append-conditioned policy,
- shared FiLM-conditioned policy,
- meta-generated / hypernetwork policy (`z_operator -> policy parameters`),
- saved policy loader,
- teacher policy,
- random policy baseline.

The shared-policy and generated-policy paths should use the same outer interface.

### Environment
- `Environment`: thin reset/step abstraction.
- `PredicateEvaluator`: raw observation -> symbolic predicate truth values / symbolic state.
- `SetupProcedure`: transforms a reset environment into the valid starting condition for a skill.
- `EnvironmentManager`: only lifecycle/orchestration where necessary (construction, reset, setup, seeding).

### Training
- `Trainer`: generic train / fine-tune abstraction.
- `TrainingBudget`: steps, episodes, gradient steps, threshold/stop conditions.
- RL algorithm adapters: SAC, PPO; BC as a separate learner / warm-start path.
- `SkillSampler`: round-robin, uniform, weighted, curriculum; enables concurrent multi-skill training without embedding task scheduling inside RL code.
- Tag experience and logs with skill/operator/run identifiers.

### Rollouts / Data
- `RolloutRunner`: run one episode for a `(skill, policy, environment)` combination and return a structured result.
- Reuse rollout logic for training, evaluation, dataset collection, visualization, and policy probing.

### Evaluation / Metrics
`EvaluationHarness` owns evaluation protocols, not training.

Required metrics:
- zero-shot success rate (per skill + aggregate),
- fine-tuning steps/time to threshold,
- fine-tuning AUC,
- transfer gain over training from scratch,
- final success rate,
- policy similarity across saved/generated/fine-tuned policies (action KL/MSE and optionally parameter-space metrics),
- component-ablation sensitivity.

Use fixed probe-state datasets for policy comparison so different policies are evaluated on identical states.

Potential future research: use component-ablation sensitivity to identify unnecessary action-model predicates and propose model pruning / repair.

### Experiments
- `ExperimentConfig`: all choices needed to reproduce a run (skills, held-out skills, seed, encoder, policy mode, learner, budgets, sampling, evaluation).
- `ExperimentRunner`: thin orchestration only: build modules from config -> train -> evaluate -> save outputs.
- factories may instantiate implementations from config, but lower-level modules must not read YAML themselves.
- support experiment sweeps.

### Artifacts / Logging
- checkpoint manager,
- structured result store (JSON/CSV),
- TensorBoard adapter,
- artifact store for configs, embeddings, models, metrics, plots,
- deterministic seed utility.

## Recommended Package Shape

```text
src/cam/
  domain/
    operators/
    parsing/
    types/
  representations/
    string/
    compositional/
      aggregators/
    graph/
  skills/
    setup/
    success/
  environments/
    observations/
    predicates/
    toy/
    fetch/
  policies/
    providers/
      shared/
      generated/
    artifacts/
  training/
    algorithms/
    sampling/
    replay/
  evaluation/
    protocols/
    metrics/
    probes/
  experiments/
  logging/
  infrastructure/

scripts/
configs/
tests/
outputs/
```

## Development Order
Build from scratch in small, reviewable steps:

1. `Predicate`, `TypedParameter`, `Effect`, `PDDLOperator`.
2. Minimal parser for current PDDL subset.
3. `SkillSpec` and setup/success interfaces.
4. Tiny toy environment + `PredicateEvaluator`.
5. `Policy` interface.
6. `OperatorEncoder` + trivial/string encoder.
7. Append-conditioned shared policy.
8. `RolloutRunner` + success metric.
9. `Trainer` + `TrainingBudget`.
10. Multi-skill `SkillSampler`.
11. Compositional aggregate encoder.
12. Slot encoder.
13. FiLM policy provider.
14. Meta-network / generated-policy provider.
15. Evaluation harness + full metric suite.
16. Fetch/MuJoCo integration.
17. Graph encoder and more advanced research extensions.

## Testing Strategy
Use TDD where practical.

- Unit-test every interface independently with fake inputs.
- Add seconds-scale toy environments before MuJoCo.
- Integration tests should cover:
  - operator -> representation -> policy -> action,
  - two-skill joint training,
  - generated policy execution,
  - fine-tuning from generated/pretrained initialization,
  - metric calculations from synthetic traces.
- Do not rely on long RL runs to discover architecture bugs.

## Key Architectural Rule
The central seams are:

```text
Task-level operator
    -> OperatorEncoder
    -> OperatorRepresentation
    -> PolicyProvider
    -> Policy
    -> Environment
    -> Rollout / Training / Evaluation
```

Everything else should support, configure, test, or measure this pipeline without tightly coupling its components.
