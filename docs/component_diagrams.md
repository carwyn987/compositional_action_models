# Component diagrams

Three views of the code as implemented: the component map (packages, entry
points, and how they connect), the data flow of one training episode, and a
class diagram. Solid boxes are implemented; dashed grey boxes are planned.
Arrows point from a component to what it uses, wraps or produces.
`docs/component_diagrams.html` renders the same diagrams in a browser.

## Component map

```mermaid
flowchart LR
    %% ---------------- entry points ----------------
    subgraph entry["Entry points (repo root / scripts)"]
        main["main.py<br/>parse_args · train · evaluate<br/>new run, resume, added skills"]
        overnight["scripts/start_overnight.sh → overnight_runs.py<br/>plan of runs for an experiment: learnability · rq1 · repair"]
        review["scripts/review_runs.py<br/>report over a folder of runs"]
        runskill["scripts/run_skill.py<br/>one skill: scripted / random / trained policy"]
        watch["scripts/watch_best_policies.py<br/>best trained policy per skill"]
        viz["scripts/visualize_environment.py<br/>render Fetch + scripted pickup"]
        tb["scripts/start_tensorboard.sh"]
        tests["run_tests.py<br/>pytest: unit / integration"]
    end

    %% ---------------- domain ----------------
    subgraph domain["cam.domain (no simulator, no numpy, no torch)"]
        direction TB
        symbols["symbols.py<br/>Predicate · TypedParameter · format_facts"]
        iface["symbolic_action_model.py<br/>SymbolicActionModel · GroundedSymbolicActionModel<br/>groundings · applicable_groundings"]
        pddl["pddl/pddl.py<br/>Precondition · Effect · PDDLOperator"]
        grounded["pddl/grounded_pddl_operator.py<br/>GroundedPDDLOperator · ground"]
        parser["pddl/pddl_parser.py<br/>parse_operator"]
        writer["pddl/pddl_writer.py<br/>to_pddl"]
        loader["action_model_library/loader.py<br/>load_symbolic_action_model"]
        files[("action_model_library/pddl/<br/>pickup · putdown · stack · unstack<br/>raised variants · place-beside · push-* .pddl")]
    end

    %% ---------------- skills ----------------
    subgraph skills["cam.skills"]
        direction TB
        skill["skill.py: Skill<br/>symbolic_action_model · setup_skills<br/>ground · applicable_groundings"]
        concrete["blocksworld_skills.py: pickup · putdown · stack · unstack<br/>+ raised variants · place-beside · pushes (4 sides, 4 diagonals, beside)"]
        registry["registry.py<br/>SKILL_REGISTRY · build_skill"]
    end

    %% ---------------- environments ----------------
    subgraph envs["cam.environments"]
        direction TB
        gymfetch["Gymnasium-Robotics Fetch"]
        multiblock["fetch/fetch_multiblock_environment.py<br/>FetchMultiBlock-v0 (1-5 blocks)"]
        annot["fetch/fetch_env_state_annotation_wrapper.py<br/>drops goal keys; info: environment_state (FetchState),<br/>objects, object_features"]
        preds["fetch/fetch_predicate_evaluation_wrapper.py<br/>FETCH_PREDICATES → info: facts<br/>check_fetch_failure · FETCH_FRAME_PREDICATES"]
        skillenv["skill_environment.py: SkillEnvironment<br/>skill per reset (round-robin / competence) · setup chain<br/>grounding · settled success · frame · failure · reward"]
        scripted["fetch/fetch_scripted_policies.py<br/>scripted pickup / pickup-raised / putdown / stack / place-beside / push"]
        fetchrewards["fetch/fetch_rewards.py<br/>staged rewards per skill (shaped)<br/>FetchMotionReward (every skill)"]
    end

    %% ---------------- policies, rewards ----------------
    subgraph policies["cam.policies / cam.rewards"]
        direction TB
        policy["policies/policy.py<br/>Policy · RandomPolicy"]
        sb3policy["policies/stable_baselines3_policy.py<br/>StableBaselines3Policy (a trained model)"]
        extractors["policies/feature_extractors.py<br/>TrainableOperatorEmbeddingExtractor"]
        reward["rewards/reward_function.py<br/>RewardFunction · SparseReward"]
    end

    %% ---------------- representations ----------------
    subgraph reps["cam.representations"]
        direction TB
        openc["operator_encoder.py: OperatorEncoder"]
        multihot["pddl_multi_hot_operator_encoder.py<br/>positional + name-only literal features"]
        onehot["one_hot_operator_encoder.py<br/>registry order, identity aliases"]
        randomenc["random_operator_encoder.py<br/>fixed random vector per name"]
        padded["padded_operator_encoder.py<br/>zero-pads to --operator-embedding-dim"]
        textenc["text_operator_encoder.py + text_backends.py<br/>PDDL or paragraph → mock / OpenAI, disk cache"]
        structure["compositional/structure.py<br/>StructuredOperatorEncoder · OperatorLayout"]
        compose["compositional/ tree · slots · geometric<br/>extractor.py: CompositionalPolicyFeaturesExtractor"]
        groundenc["grounding_encoder.py<br/>per-parameter slots"]
    end

    %% ---------------- training, evaluation, experiments ----------------
    subgraph training["cam.training"]
        direction TB
        policyobs["policy_observation_wrapper.py<br/>observation · operator_embedding · grounding"]
        sb3["stable_baselines3_trainer.py<br/>SAC / multi-task SAC / PPO · success-rate, embedding-magnitude,<br/>composition weight decay and resume warm-up callbacks"]
        mtsac["multi_task_sac.py<br/>MultiTaskSAC: an entropy temperature per skill"]
        checkpoint["checkpointing.py<br/>atomic saves · PeriodicSaveCallback"]
    end

    subgraph evaluation["cam.evaluation"]
        direction TB
        evaluate["evaluation.py<br/>evaluate: success, return, length per skill"]
        curve["learning_curve.py<br/>steps to threshold · AUC · final success"]
        metricscb["metrics_callback.py: MetricsCallback<br/>zero-shot, periodic, added-skill evaluations<br/>stop skills · patience · metrics.json"]
    end

    envsetup["cam/experiments/environment_setup.py<br/>setup_environment · OPERATOR_ENCODERS"]
    experiments["cam/experiments/research_experiments.py<br/>EXPERIMENTS: learnability · rq1 · rq3-repair"]
    logcfg["cam/logging_config.py<br/>configure_logging"]

    %% ---------------- planned ----------------
    subgraph planned["Planned"]
        direction TB
        provider["PolicyProvider: FiLM / hypernetwork"]
        learnedsetup["Setup by learned skill policies"]
    end

    %% domain
    pddl -- implements --> iface
    grounded -- implements --> iface
    pddl -.->|ground| grounded
    pddl --> symbols
    iface --> symbols
    parser --> pddl
    writer --> pddl
    loader --> parser
    loader --> files

    %% skills
    skill -- loads via --> loader
    skill -- uses --> iface
    concrete -- is-a --> skill
    registry --> concrete

    %% environment stack (inner → outer)
    multiblock -- extends --> gymfetch
    annot -- wraps --> multiblock
    preds -- wraps --> annot
    preds -- facts use --> symbols
    skillenv -- wraps --> preds
    skillenv --> skill
    skillenv -- setup chain runs --> policy
    skillenv -- reward from --> reward
    scripted -- implements --> policy
    sb3policy -- implements --> policy
    fetchrewards -- implements --> reward

    %% representations
    multihot -- implements --> openc
    onehot -- implements --> openc
    randomenc -- implements --> openc
    padded -- implements --> openc
    textenc -- implements --> openc
    structure -- implements --> openc
    padded --> multihot
    padded --> onehot
    textenc --> writer
    compose --> structure
    multihot --> pddl
    groundenc --> iface

    %% training and evaluation
    policyobs -- wraps --> skillenv
    policyobs --> openc
    policyobs --> groundenc
    sb3 -- trains on --> policyobs
    sb3 --> mtsac
    sb3 --> checkpoint
    metricscb --> evaluate
    metricscb --> curve
    metricscb --> sb3policy
    evaluate --> policy
    envsetup -- builds --> policyobs
    envsetup --> scripted
    envsetup --> fetchrewards

    %% entry points
    main --> registry
    main --> envsetup
    main --> sb3
    main -- features extractor --> extractors
    main -- features extractor --> compose
    main --> metricscb
    main --> logcfg
    overnight -- runs --> main
    overnight -- plans from --> experiments
    review -.->|reads metrics.json, TensorBoard of| metricscb
    runskill --> envsetup
    runskill --> sb3policy
    watch -- runs --> runskill
    viz -- builds --> preds
    viz --> scripted
    tb -.->|reads logs of| sb3
    tests -.-> domain
    tests -.-> skills
    tests -.-> envs
    tests -.-> reps
    tests -.-> training
    tests -.-> evaluation

    %% planned wiring
    provider -.-> sb3
    learnedsetup -.-> skillenv

    classDef plannedStyle fill:#eeeeee,stroke:#999999,stroke-dasharray:5 5,color:#666666
    class provider,learnedsetup plannedStyle
```

## One training episode

```mermaid
sequenceDiagram
    participant A as SB3 (SAC / MT-SAC / PPO)
    participant O as PolicyObservationWrapper
    participant S as SkillEnvironment
    participant F as Predicate ∘ Annotation ∘ Fetch
    participant P as setup Policy (scripted)

    A->>O: reset()
    O->>S: reset()
    S->>S: choose skill (option, next in turn, or by competence), and one of its setup chains
    S->>F: reset()
    F-->>S: obs, info{environment_state, objects, object_features, facts}
    loop each setup skill (e.g. pickup before putdown)
        S->>S: choose applicable grounding of the setup skill
        loop until its effects hold
            S->>P: action for (obs, info, grounding)
            S->>F: step(action)
            F-->>S: obs, info{facts, ...}
        end
    end
    S->>S: choose applicable grounding of the episode skill, record the start facts
    S-->>O: obs, info{skill, grounded_action_model, setup, ...}
    O-->>A: {observation, operator_embedding, grounding}
    loop each step (≤ max_steps_per_episode)
        A->>O: step(action)
        O->>S: step(action)
        S->>F: step(action)
        F-->>S: obs, info{facts, ...}
        S->>S: failure = check_failure (force, dropped, disturbed)
        S->>S: success = effects held success_hold_steps steps, frame facts unchanged, no failure
        S->>S: reward = skill reward + motion terms (− failure penalty)
        S-->>O: obs, reward, terminated = success or failure, truncated, info
        O-->>A: policy observation, reward, terminated, truncated, info{is_success, skill, ...}
    end
```

## Class diagram

Classes and modules as implemented, grouped by package (see `%%` comments).
`<<abstract>>` marks interfaces, `<<module>>` groups module-level functions and
constants, `<<external>>` marks library classes. Methods ending in `*` are abstract.
Skills, scripted policies and rewards show one or two examples each.

```mermaid
classDiagram
    direction TB

    %% ---- domain ----
    class Predicate {
        +name: str
        +args: tuple~str~
    }
    class TypedParameter {
        +name: str
        +type: str
    }
    class SymbolicActionModel {
        <<abstract>>
        +name: str
        +parameters: tuple~TypedParameter~
        +ground(binding) GroundedSymbolicActionModel*
    }
    class GroundedSymbolicActionModel {
        <<abstract>>
        +arguments: tuple~str~
        +lifted_model: SymbolicActionModel
        +preconditions_hold(facts) bool*
        +effects_hold(facts) bool*
    }
    class symbolic_action_model {
        <<module>>
        +groundings(model, objects) list
        +applicable_groundings(model, facts, objects) list
    }
    class loader {
        <<module>>
        +SYMBOLIC_ACTION_MODEL_FORMATS: dict
        +load_symbolic_action_model(skill_name, format) SymbolicActionModel
    }

    %% ---- domain.pddl ----
    class PDDLOperator {
        +preconditions: tuple~Precondition~
        +effects: tuple~Effect~
        +ground(binding) GroundedPDDLOperator
    }
    class GroundedPDDLOperator {
        +operator: PDDLOperator
        +arguments: tuple~str~
        +preconditions_hold(facts) bool
        +effects_hold(facts) bool
    }
    class Precondition {
        +predicate: Predicate
        +negated: bool
    }
    class Effect {
        +predicate: Predicate
        +delete: bool
    }
    class pddl_parser {
        <<module>>
        +parse_operator(text) PDDLOperator
    }
    class pddl_writer {
        <<module>>
        +to_pddl(operator) str
    }

    %% ---- skills ----
    class Skill {
        <<abstract>>
        +name() str*
        +setup_skills: tuple~tuple~str~~
        +symbolic_action_model: SymbolicActionModel
        +ground(binding) GroundedSymbolicActionModel
        +applicable_groundings(facts, objects) list
    }
    class PickupSkill
    class PutdownSkill {
        +setup_skills = (pickup), (pickup-raised)
    }
    class registry {
        <<module>>
        +SKILL_REGISTRY: dict
        +build_skill(skill_name, config) Skill
    }

    %% ---- policies, rewards ----
    class Policy {
        <<abstract>>
        +reset()
        +__call__(obs, info, grounded) ndarray*
    }
    class RandomPolicy
    class StableBaselines3Policy {
        +model: BaseAlgorithm
        +deterministic: bool
        +load(model_path, algorithm)$
    }
    class RewardFunction {
        <<abstract>>
        +reset(grounded, info)
        +__call__(grounded, action, info, success) tuple*
    }
    class SparseReward

    %% ---- environments ----
    class GymWrapper {
        <<external>>
    }
    class MujocoFetchEnv {
        <<external>>
    }
    class FetchMultiBlockEnv {
        +num_blocks: int
    }
    class FetchState {
        +gripper_position: ndarray
        +finger_width: float
        +block_positions: dict
        +block_rotations: dict
        +gripper_force: ndarray
    }
    class FetchEnvStateAnnotationWrapper {
        +extract_state(obs) FetchState
    }
    class FetchPredicateEvaluationWrapper {
        +evaluate(state) frozenset~Predicate~
    }
    class fetch_predicates {
        <<module>>
        +FETCH_PREDICATES: dict
        +FETCH_PREDICATE_ARITIES: dict
        +FETCH_FRAME_PREDICATES: frozenset
        +check_fetch_failure(grounded, start_info, info) str
    }
    class SkillEnvironment {
        +skills: dict
        +setup_chains: dict
        +setup_policies: dict
        +reward_functions: dict
        +shared_reward: RewardFunction
        +check_failure: callable
        +frame_predicates: frozenset
        +success_hold_steps: int
        +competence_sampling: bool
        +competence_shares() dict
        +reset(seed, options) tuple
        +step(action) tuple
    }
    class FetchScriptedPickupPolicy
    class FetchScriptedPushPolicy
    class FetchPickupReward
    class FetchMotionReward

    %% ---- representations ----
    class OperatorEncoder {
        <<abstract>>
        +dim() int*
        +encode(model) ndarray*
    }
    class PDDLMultiHotOperatorEncoder {
        +features: list
        +max_arity: int
    }
    class OneHotOperatorEncoder {
        +index: dict
        +aliases: dict
    }
    class RandomOperatorEncoder {
        +seed: int
        +aliases: dict
    }
    class PaddedOperatorEncoder {
        +encoder: OperatorEncoder
    }
    class TextOperatorEncoder
    class TextEmbeddingBackend {
        <<abstract>>
    }
    class MockTextBackend
    class OpenAIBackend
    class CachedTextBackend
    class StructuredOperatorEncoder {
        +layout: OperatorLayout
    }
    class OperatorLayout
    class GroundingEncoder {
        +max_arity: int
        +max_objects: int
        +encode(grounded, objects, object_features) ndarray
    }

    %% ---- policy features extractors (trained with the policy) ----
    class BaseFeaturesExtractor {
        <<external>>
    }
    class TrainableOperatorEmbeddingExtractor {
        +operator_embedding: Linear
    }
    class CompositionalPolicyFeaturesExtractor {
        +composition: Module
    }
    class TreeComposition
    class SlotComposition
    class GeometricComposition

    %% ---- training ----
    class PolicyObservationWrapper {
        +operator_encoder: OperatorEncoder
        +grounding_encoder: GroundingEncoder
    }
    class main {
        <<module>>
        +train(config, env, skills, run_directory, resume)
    }
    class stable_baselines3_trainer {
        <<module>>
        +train_stable_baselines3(env, algorithm, ...) model
    }
    class SAC {
        <<external>>
    }
    class MultiTaskSAC {
        +skill_slots: dict
        +skill_temperatures() dict
    }
    class BaseCallback {
        <<external>>
    }
    class SuccessRateCallback
    class OperatorEmbeddingMagnitudeCallback
    class CompositionWeightDecayCallback
    class ResumeWarmupCallback
    class PeriodicSaveCallback

    %% ---- evaluation ----
    class MetricsCallback {
        +skill_names: list
        +eval_interval: int
        +stop_skills: list
    }
    class evaluation {
        <<module>>
        +evaluate(env, policy, skills, episodes) EvaluationResult
    }
    class LearningCurve

    %% inheritance
    SymbolicActionModel <|-- PDDLOperator
    GroundedSymbolicActionModel <|-- GroundedPDDLOperator
    Skill <|-- PickupSkill
    Skill <|-- PutdownSkill
    Policy <|-- RandomPolicy
    Policy <|-- StableBaselines3Policy
    Policy <|-- FetchScriptedPickupPolicy
    Policy <|-- FetchScriptedPushPolicy
    RewardFunction <|-- SparseReward
    RewardFunction <|-- FetchPickupReward
    RewardFunction <|-- FetchMotionReward
    MujocoFetchEnv <|-- FetchMultiBlockEnv
    GymWrapper <|-- FetchEnvStateAnnotationWrapper
    GymWrapper <|-- FetchPredicateEvaluationWrapper
    GymWrapper <|-- SkillEnvironment
    GymWrapper <|-- PolicyObservationWrapper
    OperatorEncoder <|-- PDDLMultiHotOperatorEncoder
    OperatorEncoder <|-- OneHotOperatorEncoder
    OperatorEncoder <|-- RandomOperatorEncoder
    OperatorEncoder <|-- PaddedOperatorEncoder
    OperatorEncoder <|-- TextOperatorEncoder
    OperatorEncoder <|-- StructuredOperatorEncoder
    TextEmbeddingBackend <|-- MockTextBackend
    TextEmbeddingBackend <|-- OpenAIBackend
    TextEmbeddingBackend <|-- CachedTextBackend
    BaseFeaturesExtractor <|-- TrainableOperatorEmbeddingExtractor
    BaseFeaturesExtractor <|-- CompositionalPolicyFeaturesExtractor
    BaseCallback <|-- SuccessRateCallback
    BaseCallback <|-- OperatorEmbeddingMagnitudeCallback
    BaseCallback <|-- CompositionWeightDecayCallback
    BaseCallback <|-- ResumeWarmupCallback
    BaseCallback <|-- PeriodicSaveCallback
    SAC <|-- MultiTaskSAC
    BaseCallback <|-- MetricsCallback

    %% composition / aggregation
    SymbolicActionModel *-- TypedParameter
    PDDLOperator *-- Precondition
    PDDLOperator *-- Effect
    Precondition *-- Predicate
    Effect *-- Predicate
    GroundedPDDLOperator o-- PDDLOperator
    Skill o-- SymbolicActionModel
    SkillEnvironment o-- Skill
    SkillEnvironment o-- Policy : setup
    SkillEnvironment o-- RewardFunction
    SkillEnvironment o-- GroundedSymbolicActionModel : episode
    FetchPredicateEvaluationWrapper o-- FetchEnvStateAnnotationWrapper : wraps
    SkillEnvironment o-- FetchPredicateEvaluationWrapper : wraps
    PolicyObservationWrapper o-- SkillEnvironment : wraps
    PolicyObservationWrapper o-- OperatorEncoder
    PolicyObservationWrapper o-- GroundingEncoder
    PaddedOperatorEncoder o-- OperatorEncoder
    TextOperatorEncoder o-- TextEmbeddingBackend
    CachedTextBackend o-- TextEmbeddingBackend
    StructuredOperatorEncoder o-- OperatorLayout
    CompositionalPolicyFeaturesExtractor o-- TreeComposition
    CompositionalPolicyFeaturesExtractor o-- SlotComposition
    CompositionalPolicyFeaturesExtractor o-- GeometricComposition
    MetricsCallback o-- LearningCurve

    %% dependencies
    SymbolicActionModel ..> GroundedSymbolicActionModel : ground()
    symbolic_action_model ..> SymbolicActionModel
    loader ..> pddl_parser
    pddl_parser ..> PDDLOperator : creates
    pddl_writer ..> PDDLOperator : reads
    Skill ..> loader
    Skill ..> symbolic_action_model
    registry ..> PickupSkill
    registry ..> PutdownSkill
    FetchEnvStateAnnotationWrapper ..> FetchState : creates
    FetchPredicateEvaluationWrapper ..> fetch_predicates
    FetchPredicateEvaluationWrapper ..> Predicate : creates
    SkillEnvironment ..> fetch_predicates : failure, frame
    PDDLMultiHotOperatorEncoder ..> PDDLOperator : encodes
    TextOperatorEncoder ..> pddl_writer
    StructuredOperatorEncoder ..> PDDLOperator : encodes
    CompositionalPolicyFeaturesExtractor ..> OperatorLayout : unpacks
    GroundingEncoder ..> GroundedSymbolicActionModel : encodes
    stable_baselines3_trainer ..> PolicyObservationWrapper : trains on
    stable_baselines3_trainer ..> SuccessRateCallback
    stable_baselines3_trainer ..> PeriodicSaveCallback
    stable_baselines3_trainer ..> ResumeWarmupCallback
    stable_baselines3_trainer ..> MultiTaskSAC : mtsac
    main ..> stable_baselines3_trainer
    main ..> TrainableOperatorEmbeddingExtractor : policy_kwargs
    main ..> CompositionalPolicyFeaturesExtractor : policy_kwargs
    main ..> OperatorEmbeddingMagnitudeCallback
    main ..> CompositionWeightDecayCallback
    main ..> MetricsCallback
    MetricsCallback ..> evaluation
    MetricsCallback ..> StableBaselines3Policy
    evaluation ..> Policy
```
