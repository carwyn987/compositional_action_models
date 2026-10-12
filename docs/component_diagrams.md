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
        main["main.py<br/>parse_args · setup_environment<br/>train · run_random_actions"]
        runskill["scripts/run_skill.py<br/>one skill, scripted / random policy"]
        viz["scripts/visualize_environment.py<br/>render Fetch + scripted pickup"]
        tb["scripts/start_tensorboard.sh"]
        tests["run_tests.py<br/>pytest: unit / integration"]
    end

    %% ---------------- domain ----------------
    subgraph domain["cam.domain (no simulator, no torch)"]
        direction TB
        symbols["symbols.py<br/>Predicate · TypedParameter · format_facts"]
        iface["symbolic_action_model.py<br/>SymbolicActionModel · GroundedSymbolicActionModel<br/>groundings · applicable_groundings"]
        pddl["pddl/pddl.py<br/>Precondition · Effect · PDDLOperator"]
        grounded["pddl/grounded_pddl_operator.py<br/>GroundedPDDLOperator · ground"]
        parser["pddl/pddl_parser.py<br/>parse_operator"]
        writer["pddl/pddl_writer.py<br/>to_pddl"]
        loader["action_model_library/loader.py<br/>load_symbolic_action_model"]
        files[("action_model_library/pddl/<br/>pickup · putdown · stack · unstack<br/>pickup-raised · unstack-raised · place-beside .pddl")]
    end

    %% ---------------- skills ----------------
    subgraph skills["cam.skills"]
        direction TB
        skill["skill.py: Skill<br/>symbolic_action_model · setup_skills<br/>ground · applicable_groundings"]
        concrete["blocksworld_skills.py: pickup · putdown · stack · unstack<br/>+ pickup-raised · unstack-raised · place-beside"]
        registry["registry.py<br/>SKILL_REGISTRY · build_skill"]
    end

    %% ---------------- environments ----------------
    subgraph envs["cam.environments"]
        direction TB
        gymfetch["Gymnasium-Robotics Fetch"]
        multiblock["fetch/fetch_multiblock_environment.py<br/>FetchMultiBlock-v0 (1-5 blocks)"]
        annot["fetch/fetch_env_state_annotation_wrapper.py<br/>drops goal keys; info: environment_state,<br/>objects, object_features"]
        preds["fetch/fetch_predicate_evaluation_wrapper.py<br/>FETCH_PREDICATES → info: facts"]
        skillenv["skill_environment.py: SkillEnvironment<br/>skill per reset · setup chain · grounding<br/>success · reward"]
        scripted["fetch/fetch_scripted_policies.py<br/>scripted pickup / putdown / stack / place-beside"]
        fetchrewards["fetch/fetch_rewards.py<br/>staged rewards for every skill (shaped)"]
    end

    %% ---------------- policies, rewards ----------------
    subgraph policies["cam.policies / cam.rewards"]
        direction TB
        policy["policies/policy.py<br/>Policy · RandomPolicy"]
        reward["rewards/reward_function.py<br/>RewardFunction · SparseReward"]
    end

    %% ---------------- representations ----------------
    subgraph reps["cam.representations (numpy)"]
        direction TB
        openc["operator_encoder.py: OperatorEncoder"]
        multihot["pddl_multi_hot_operator_encoder.py<br/>positional + name-only literal features"]
        onehot["one_hot_operator_encoder.py<br/>baseline"]
        groundenc["grounding_encoder.py<br/>per-parameter slots"]
    end

    %% ---------------- training ----------------
    subgraph training["cam.training"]
        direction TB
        policyobs["policy_observation_wrapper.py<br/>observation · operator_embedding · grounding"]
        sb3["stable_baselines3_trainer.py<br/>SAC / PPO · SuccessRateCallback"]
    end

    logcfg["cam/logging_config.py<br/>configure_logging"]

    %% ---------------- planned ----------------
    subgraph planned["Planned"]
        direction TB
        learnedenc["Learned / aggregating operator encoders"]
        evalh["Evaluation harness (zero-shot, fine-tuning metrics)"]
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
    fetchrewards -- implements --> reward

    %% representations
    multihot -- implements --> openc
    onehot -- implements --> openc
    multihot --> pddl
    groundenc --> iface

    %% training
    policyobs -- wraps --> skillenv
    policyobs --> openc
    policyobs --> groundenc
    sb3 -- trains on --> policyobs

    %% entry points
    main --> registry
    main -- builds --> policyobs
    main --> sb3
    main --> logcfg
    runskill -- builds --> skillenv
    runskill --> scripted
    viz -- builds --> preds
    viz --> scripted
    tb -.->|reads logs of| sb3
    tests -.-> domain
    tests -.-> skills
    tests -.-> envs
    tests -.-> reps
    tests -.-> training

    %% planned wiring
    learnedenc -.-> openc
    evalh -.-> skillenv
    learnedsetup -.-> skillenv

    classDef plannedStyle fill:#eeeeee,stroke:#999999,stroke-dasharray:5 5,color:#666666
    class learnedenc,evalh,learnedsetup plannedStyle
```

## One training episode

```mermaid
sequenceDiagram
    participant A as SB3 (SAC / PPO)
    participant O as PolicyObservationWrapper
    participant S as SkillEnvironment
    participant F as Predicate ∘ Annotation ∘ Fetch
    participant P as setup Policy (scripted)

    A->>O: reset()
    O->>S: reset()
    S->>S: choose skill (option or next in turn)
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
    S->>S: choose applicable grounding of the episode skill
    S-->>O: obs, info{skill, grounded_action_model, setup, ...}
    O-->>A: {observation, operator_embedding, grounding}
    loop each step (≤ max_steps_per_episode)
        A->>O: step(action)
        O->>S: step(action)
        S->>F: step(action)
        F-->>S: obs, info{facts, ...}
        S->>S: success = effects_hold(facts), reward = RewardFunction(...)
        S-->>O: obs, reward, terminated=success, truncated, info
        O-->>A: policy observation, reward, terminated, truncated, info{is_success, skill, ...}
    end
```

## Class diagram

Classes and modules as implemented, grouped by package (see `%%` comments).
`<<abstract>>` marks interfaces, `<<module>>` groups module-level functions and
constants, `<<external>>` marks library classes. Methods ending in `*` are abstract.

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
        +setup_skills: tuple~str~
        +symbolic_action_model: SymbolicActionModel
        +ground(binding) GroundedSymbolicActionModel
        +applicable_groundings(facts, objects) list
    }
    class PickupSkill
    class PutdownSkill {
        +setup_skills = pickup
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
    }
    class FetchEnvStateAnnotationWrapper {
        +num_blocks: int
        +extract_state(obs) FetchState
    }
    class FetchPredicateEvaluationWrapper {
        +evaluate(state) frozenset~Predicate~
    }
    class fetch_predicates {
        <<module>>
        +FETCH_PREDICATES: dict
        +FETCH_PREDICATE_ARITIES: dict
    }
    class SkillEnvironment {
        +skills: dict
        +setup_policies: dict
        +reward_functions: dict
        +grounded_action_model: GroundedSymbolicActionModel
        +reset(seed, options) tuple
        +step(action) tuple
    }
    class FetchScriptedPickupPolicy
    class FetchScriptedPutdownPolicy
    class FetchPickupReward {
        +config: FetchPickupRewardConfig
    }

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
    class OneHotOperatorEncoder
    class GroundingEncoder {
        +max_arity: int
        +max_objects: int
        +encode(grounded, objects, object_features) ndarray
    }

    %% ---- training ----
    class PolicyObservationWrapper {
        +operator_encoder: OperatorEncoder
        +grounding_encoder: GroundingEncoder
    }
    class stable_baselines3_trainer {
        <<module>>
        +train_stable_baselines3(env, algorithm, total_timesteps, run_directory) model
    }
    class SuccessRateCallback

    %% inheritance
    SymbolicActionModel <|-- PDDLOperator
    GroundedSymbolicActionModel <|-- GroundedPDDLOperator
    Skill <|-- PickupSkill
    Skill <|-- PutdownSkill
    Policy <|-- RandomPolicy
    Policy <|-- FetchScriptedPickupPolicy
    Policy <|-- FetchScriptedPutdownPolicy
    RewardFunction <|-- SparseReward
    RewardFunction <|-- FetchPickupReward
    MujocoFetchEnv <|-- FetchMultiBlockEnv
    GymWrapper <|-- FetchEnvStateAnnotationWrapper
    GymWrapper <|-- FetchPredicateEvaluationWrapper
    GymWrapper <|-- SkillEnvironment
    GymWrapper <|-- PolicyObservationWrapper
    OperatorEncoder <|-- PDDLMultiHotOperatorEncoder
    OperatorEncoder <|-- OneHotOperatorEncoder

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
    PDDLMultiHotOperatorEncoder ..> PDDLOperator : encodes
    GroundingEncoder ..> GroundedSymbolicActionModel : encodes
    stable_baselines3_trainer ..> PolicyObservationWrapper : trains on
    stable_baselines3_trainer ..> SuccessRateCallback
```
