# Project status

What the code does today and what the experiments so far show. The research questions are in
`docs/research_focus.md`; how to run each experiment is in `docs/experiments.md`; the backlog (issues,
features, what to do next) is in `docs/next_steps.md`.

## 1. Environment and skills

### Pipeline
Fetch simulator (`FetchMultiBlock-v0`, 1–5 blocks), wrapped from the inside out:
1. `FetchEnvStateAnnotationWrapper`: named state, including block positions and rotations.
2. `FetchPredicateEvaluationWrapper`: the facts true in the current state.
3. `SkillEnvironment`: per episode, picks a skill (`--skill-sampling`, §2), runs one of its setup chains with
   scripted policies, picks a random applicable grounding that is feasible in the scene (a push only with no
   other block on its path, `push_path_clear`: otherwise it could only fail, by knocking that block aside), and
   computes success, failure and reward.
4. `TimeLimit`: outside `SkillEnvironment`, so setup steps don't count toward the episode's limit
   (`--max-steps-per-episode`, default 100).
5. `PolicyObservationWrapper`: builds `{observation, operator_embedding, grounding}` (`docs/policy_inputs.md`).

`docs/component_diagrams.md` maps every module and class.

Design rules the code keeps:
- `cam.domain` (symbols, action models, the PDDL parser and writer) imports neither Gymnasium, NumPy nor
  torch.
- Action models sit behind `SymbolicActionModel` / `GroundedSymbolicActionModel` and the loader's format
  table (`--symbolic-action-model-format`, PDDL only), so another action-model format plugs in without
  changing skills or environments.
- Small interfaces with swappable implementations: `SymbolicActionModel`, `OperatorEncoder`, `Policy`,
  `RewardFunction`.
- Facts are closed-world and use the action models' predicate names and arities. They follow the PDDL
  meaning, not physical intuition: a held block is not `clear`.
- Predicates are functions of a `FetchState` (named fields), so they are unit-tested on hand-built states
  without the simulator.
- Setup acts through the normal `step` API, never by setting the simulator state, so every set-up state
  is physically reachable.
- Evaluation runs on its own environment, so training episodes are not interrupted.
- A run is reproducible from its `config.json` (every command-line option); options that would change the
  policy's observation or algorithm cannot change on resume (`LOCKED_ON_RESUME` in `main.py`).

### Predicates (`fetch_predicate_evaluation_wrapper.py`)
Block positions are block **centres**. Blocks are 5 cm cubes, so a block resting on the table has its
centre at z = 0.425, and a block stacked on another has its centre at 0.475.

| predicate | true when |
| --- | --- |
| `holding(b)` | gripped (fingers closed around it) **and** lifted off its support (table or block) |
| `on-table(b)` | centre within 1 cm of table rest height |
| `on(b, c)` | b rests on c: within half a block horizontally, one block height above it |
| `clear(b)` | nothing on b and not held |
| `gripper-empty` | no block gripped (fingers open, or closed on nothing) |
| `raised(b)` | held, centre **30–36 cm** above table rest: a band, so the block must be stopped there, not just passed (25 cm of lift from a stacked start) |
| `beside(b, c)` | both on the table, centres 5.5–10 cm apart (reset places blocks ≥ 10 cm apart) |
| `at-left/right/front/back(b)` | on the table, **10–18 cm** from the centre of the reset area (1.34, 0.75) in that direction (front = toward the robot, left = +y) |

The regions are global, so pushes have fixed goals.

### Success, failure and motion penalties
**Success is settled and respects the frame:** the effects must hold for `--success-hold-steps` (default 5)
consecutive steps with the moved block settled over them (rising or sinking less than 1 cm net,
`SETTLE_DISTANCE`), so a skill must stop in its goal state: a block lifted on and on fast enough to reach the
raised band within an episode has not stopped in it, while an exploring policy's jitter mostly cancels out; and
every fact the action does not mention must keep its value from the episode's start (the STRIPS frame
assumption, checked over `on-table`, `on`, `clear`, `holding`, `gripper-empty`; the geometric predicates change
as side effects and are left out). So unstack(?o ?u) must leave `(on-table ?u)` true: lifting both blocks,
which a trained policy did, is not an unstack.

**Failure** ends an episode unsuccessfully, with a penalty of 2 under shaped rewards, and takes precedence
over success. It is the same for every skill (`check_fetch_failure`): an **extreme force** on the gripper (net
contact force above `MAX_GRIPPER_FORCE`, 500 N, for `FORCE_FAILURE_STEPS`, 5, steps in a row: pressing on a
block or the table), the moved block (first argument) was **dropped** (let go more than `DROP_HEIGHT`,
2 cm, above its support), or another block was **disturbed** (moved more than `DISTURB_DISTANCE`, 3 cm,
from where it started: knocked or pushed aside). The episode ends on the failing step, before the failure plays out
(a block let go 20 cm up has not fallen yet), so on that step the shaped reward also charges back the moved block's
progress of the episode (the `refund` component: lift, push, carry, align, descent); the stage bonuses and the
gripper's approach are kept, since failing cannot earn them. Without the refund, lifting a block and letting it go
kept the lift's reward.

**Motion penalties** are added to every skill's shaped reward (`FetchMotionReward`), against brutal motion
(without them, trained push policies threw blocks): action changes (smoothness), the moved block's
acceleration, any movement of the other blocks, and the **gripper force** above `SAFE_GRIPPER_FORCE`
(100 N), counted up to 500 N: at most 0.2 per step, so pressing until it fails costs half as much as the failure.
The gripper force is the net contact force on the gripper and fingers, read from MuJoCo
(`FetchState.gripper_force`, with the last 10 steps' magnitudes in `gripper_force_history`): a grasp's
squeeze cancels out, so it measures pressing on a block or the table,
pushing, ramming, and a carried block's weight (~20 N). Lifting and placing peak around 55 N, steady pushing
around 65 N; pressing into the table stays around 55 kN. First contact spikes for a single step (ramming a
block with closed fingers: ~900 N once; random exploration: up to ~38 kN on about 1% of steps), and random
exploration bumps the table for a few steps at a time, which is why the penalty is capped and failure needs 5
steps. Force limits matter for moving to real robots, whose grippers such forces would damage.

Grasping skills therefore succeed only once the block is lifted and held, and placing skills only once it is
released, gently, without disturbing anything else.

The motion penalties are light on purpose: they grow with exploration noise wherever the gripper touches
something, and with ten times these weights (2026-10-14 run) an exploring policy did better keeping away from the
blocks than trying (scripted pickup with noise σ = 0.6: −7.4 against −0.2 for doing nothing; now −1.9 against
−4.2). The failure penalty is small for the same reason: exploring policies fail often near the blocks (a noisy
finger command lets a lifted block go, a noisy height command presses the fingers into the table), and at 10 they
learned to keep away: pickup-raised trained alone hovered over its block with the fingers closed, 0 of 10 at every
evaluation up to 60k steps. Every shaped reward also costs 0.05 per step, at most 1 from any step on at the
training's discount (γ = 0.95): waiting is not free, and costs less than a failure, so failing never pays.

### Skills (`skills/blocksworld_skills.py`, PDDL in `domain/action_model_library/pddl/`)

| skill | operator | setup | scripted policy | shaped reward |
| --- | --- | --- | --- | --- |
| pickup(?o) | classic blocksworld + `(not (raised ?o))`: hold it low | – | approach, descend, close, lift | staged pickup, ending with a settle bonus (held at rest, below the raised band) |
| putdown(?o) | classic | pickup **or** pickup-raised (random per episode) | lower, open | staged putdown (gentle descent, square) |
| stack(?o ?u) | classic | pickup | carry up, align, lower, open | staged stack |
| unstack(?o ?u) | classic + `(not (raised ?o))` | pickup, stack, then the open gripper moved away (up 6–12 cm, then 5–15 cm sideways in a random direction), so it starts with an approach | pickup motion | pickup's stages, lift from the block below |
| pickup-raised(?o) | pickup + `(raised ?o)` | – | pickup motion to 33 cm | pickup's stages, lift peaking at 33 cm (overshooting costs), settle bonus in the band |
| unstack-raised(?o ?u) | unstack + `(raised ?o)` | as unstack | pickup motion to 33 cm | unstack's stages, lift peaking 28 cm above the base rest (overshooting costs), settle bonus in the band |
| place-beside(?o ?u) | putdown + ?u + `(beside ?o ?u)` | pickup | stack motion, landing next to ?u along x | stack's stages, target ring around ?u |
| push-left/right/front/back(?o) | pre `on-table`, `clear`, `gripper-empty`, `(not (at-X ?o))`; eff `(at-X ?o)` | – | rise, behind the block, lower, push toward the band's middle | over the block to the point behind it, down to it and staying there as the block moves, push progress |
| push-right-front, push-left-back, push-right-back, push-left-front(?o) | as push, both regions | – | two straight pushes | as push |
| push-beside(?o ?u) | as push, + `(on-table ?u)`; `beside` in place of the region | – | push toward ?u | as push |

Which skills are trained and which are held out is set by the experiment (§4), not by the skill definitions:
in `rq3-repair-minimal` (`src/cam/experiments/research_experiments.py`), seven skills train and unstack-raised,
place-beside and push-left-front are held out; in `rq3-repair-maximal`, the 13 training skills (every push but
push-left-front) train and push-left-front is held out.

Every scripted policy succeeds with 5-step settled success (2 and 3 blocks): pick/place skills 10/10, pushes
40/40 (2 blocks). Without the clear-path rule another block lies on the path in a fifth of push-left's scenes and
two thirds of a diagonal's, and the scripted push-left-front then succeeds in 30/40 (knocking the other block in
9). Diagonal scenes are resampled until the path is clear, about four resets each. The scripted push moves at a quarter
of full speed while pushing (`PUSH_SPEED`), which keeps its contact force low; a diagonal push then takes
about 30–120 steps (mostly 70–100), so the pre-flight check gives scripted policies 150 steps.

## 2. Policy and training

### Operator embedding conditions (`--operator-encoder`, `--operator-embedding-dim 256`)
`docs/operator_embeddings.md` describes each with worked examples; `docs/research_focus.md` lists them as the
experimental conditions.
- **Identity only:** one-hot (indexed by registry order, so indices don't depend on the training
  skills); random (a fixed vector per name).
- **Structure, nothing learned:** multi-hot (192 literal features); text (a paragraph or the PDDL,
  embedded by OpenAI `text-embedding-3-small` or the offline mock, which hashes character n-grams and so only
  matches shared snippets of text).
- **Learned:** `--trainable-operator-embedding` (a learnable linear layer on top); compositional (tree /
  slots / geometric, composed inside the policy, normalized to magnitude 1, with composition weight decay).
- `--operator-identity-aliases NEW=OLD` makes one-hot and random encode NEW exactly as OLD. This is the
  repair-hidden control. Every other operator keeps its code (one-hot keeps the indices it has without
  aliases, leaving NEW's unused), so the other skills of a repair-hidden run keep the codes of the rq1 run
  it continues.
- **The operator's name** is in the text embedding's text and is a component of the compositional embedding,
  unless left out (`--operator-text-name none`, `--compositional-name none`): held-out operators share words
  of their names with training skills, so the experiments leave it out (`OPERATOR_NAMES`, below).

**Compositional embedding scale.** The compositional extractor (`representations/compositional/extractor.py`)
normalizes the composed embedding to zero mean and magnitude (L2 norm) 1 before the policy sees it. Without
the normalization the tree's embedding was about 100–500× larger than the policy's other inputs, and tree
never learned pickup (§5, 2026-10-08). The raw output before normalization has no fixed scale: about 50–150
for tree and 20–30 for slots and geometric at initialization, and it grows during training (slots: from
about 2.5 to 800–2800 over 500k SAC steps in the 2026-10-10 run). Since the normalized output does not depend
on that scale, nothing in the loss pulls it back, and as the weights grow, each Adam step (of roughly constant
size) changes the normalized output less: the composition learns more slowly late in training, by an amount
set by each layer's weight growth (not measured).

`--composition-weight-decay` (default 0.01, compositional runs only; `CompositionWeightDecayCallback`) is
decoupled weight decay, as in AdamW, on the composition alone: after each gradient update, every parameter
under each features extractor's `composition` (actor and critic for SAC, the shared one for PPO) is
multiplied by (1 − lr·wd). Weight decay shrinks every parameter toward zero and the learning signal re-grows
what it needs, so the scale settles where the two balance. Too strong, and the composition underfits and
operators blur together; it also shrinks the embeddings of rarely seen predicates, which get few updates to
recover. Tune wd by watching `operator_embedding_magnitude_before_normalization/*` in TensorBoard: it should
settle at O(1–10) instead of growing. Other remedies, should it not, are in `docs/next_steps.md`.

### Algorithms and skill sampling
Training is Stable-Baselines3 SAC (`main.py`'s default), multi-task SAC (`--algorithm mtsac`, the overnight
runner's default) or PPO, with discount 0.95 and, for SAC, one gradient update per environment step and a
300k-transition replay buffer (`HYPERPARAMETERS` in `stable_baselines3_trainer.py`). Features extractors
compute the trainable and compositional embeddings inside the policy. Multi-task SAC (`MultiTaskSAC`) keeps a
separate entropy temperature per skill, as on Meta-World: with one temperature, the entropy target holds only
on average over the skills, so learned skills and skills still being explored pull it in opposite directions.
Each distinct operator embedding gets its own temperature, logged as `train/ent_coef/<skill>`; the policy's
input is unchanged.

**Skill sampling** (`--skill-sampling`): **round-robin** (the skills take turns) or **competence**: each skill
keeps half its even share of the episodes, and the rest goes to skills in proportion to their recent failure
rate (over their last 20 training episodes; an episode cut off by the time limit is a failure), so skills
that are learned give their episodes to skills that are not. The choice is a smooth weighted round-robin, so
the shares hold without sampling noise. The overnight runner uses competence for every condition. A skill
that cannot be learned keeps a large share, so gate such skills out of the training set.

## 3. Evaluation

### Metrics (`evaluation/`)
Evaluated every `--eval-interval` steps (50k) with `--eval-episodes` (20) per skill, deterministic and
stochastic, on a separate environment reseeded identically each time. A skill added on resume (a repaired or
new operator) is also evaluated alone 1k, 2k, 4k, … steps after it was added, doubling until
`--eval-interval` takes over: precise steps to threshold early, where conditions differ most, at little cost
(`ADDED_SKILL_FIRST_EVALUATION` in `metrics_callback.py`). Results go to TensorBoard and `<run>/metrics.json`.
- **success rate**;
- **steps / episodes to threshold:** from the skill's first evaluation, until success ≥ 0.95
  (`--success-threshold`), counted in all training steps and in the skill's own (**on skill**,
  `skill_steps_to_threshold`). Skills share a run's steps, so the skill's own steps are what compare across
  skills, across ways of sampling skills, and with learning the skill alone;
- **zero-shot:** the first evaluation of a skill;
- **final success**;
- **AUC:** the *average* success rate over the run (area under the curve divided by the steps). It stays
  low if early success is low, even when the run ends at 100%. Compare AUC only between runs with the same
  budget;
- **retention** (after a skill is added): for every other skill, its level before (the mean of its last 2
  evaluations), its lowest and last success after, and when it recovered (back within 0.1 of its level on 2
  evaluations in a row, counting only evaluations after the addition): how much the new skill made the
  policy forget (`"retention"` in `metrics.json`).

Over seeds (the runs of a condition that differ only in seed), `scripts/review_runs.py` reports each metric's mean
± std and, from 2 seeds, a percentile bootstrap 95% confidence interval of the mean (10,000 resamples from a fixed
seed, so a folder always gives the same report). With 3 seeds or fewer the interval is the seeds' range, which holds
the true mean less often than 95% (about 3 times in 4 with 3 seeds, about 9 in 10 with 10). A seed that never reached
the threshold, or whose other skills never recovered, counts as what it trained: left out, it would make a
condition that often fails look fast; counted so, the mean, its interval and the transfer gain are lower bounds,
marked `≥` and followed by the number of such seeds.

Training also logs each skill's training success rate (`success_rate/<skill>`) and, every 1000 steps, its
operator embedding's magnitude (`operator_embedding_magnitude/*`, and for compositional runs
`..._before_normalization/*`).

### Repair modes
`--repair-mode`, resuming with the repaired skill in `--stop-skills`:
- **isolated:** only the repaired skill trains; the others are evaluated, not trained. Stops once it is at
  the threshold on 2 evaluations in a row. Its steps compare directly with learning the skill from scratch.
- **retain:** every skill trains, each episode's skill chosen as in the run it continues. Stops once the
  repaired skill is at the threshold and every other skill has recovered (`--recovery-tolerance`, 0.1): the
  cost of a repair without forgetting.

## 4. Scripts
- **`scripts/start_overnight.sh`:** checks the environment (ignores the shell's `PYTHONPATH`, e.g. ROS),
  runs the unit tests, writes `run_info.txt` (commit and settings), blocks sleep, then runs
  `overnight_runs.py`.
- **`scripts/overnight_runs.py`:** runs an experiment's plan of runs.
  - **experiments** (`--experiment NAME`, default `rq3-repair-minimal`; `--list` describes each) are classes in
    `src/cam/experiments/research_experiments.py`: the docstring gives the question, design, controls and what
    to read; the attributes are the runner's settings (groups, skills, repairs, conditions, budgets). An
    environment variable named as a setting overrides it, and the runner lists the overrides. `learnability`:
    each registered skill alone; `rq1`: one shared policy per operator embedding on the training skills;
    `rq3-repair-minimal` and `rq3-repair-maximal`: rq1's policies continued with repaired operators (few training
    skills, each repair's new literals in one other skill; or every training skill and one repair); `overlap`: how much of their operators'
    structure the training skills share, at five levels from predicates to whole operators, and how much of each
    held-out operator they cover (`src/cam/experiments/representation_overlap.py`; static, nothing trains). Every
    experiment prints the same report for its own skills when it starts and saves it as `<OUT>/overlap.txt`;
  - **refuses repairs that test nothing learnable:** each repaired operator must be held out, replace a training
    skill, and have every literal (place in the operator, sign, predicate, arguments) in some training skill;
  - **groups:** learnability (each skill alone, with `LEARNABILITY_EMBEDDING` (multi-hot), evaluated every `LEARN_EVAL_INTERVAL` (10k) steps: its
    steps to threshold are the from-scratch baseline and set `RQ1_STEPS=auto`; also the individually trained
    baseline for RQ4),
    rq1 (one shared policy per condition and seed on `BASE_SKILLS`), repair (each `FROM:TO` in `REPAIRS`, in
    each of `REPAIR_MODES`, repair-visible and repair-hidden; every skill evaluated every
    `REPAIR_EVAL_INTERVAL` (10k) steps);
  - **defaults** (the `Experiment` base class, and every run's options): multi-task SAC, competence sampling,
    shaped rewards, 2 blocks, the OpenAI text backend, the operator's name left out of the text and compositional
    embeddings (`OPERATOR_NAMES=exclude`, runs labelled `<condition>-no-name`; `OPERATOR_NAMES="exclude include"`
    also runs them with it, the ablation), 13 training skills; the `TO` of each repair pair
    (`unstack-raised`, `place-beside`, `push-left-front`) is held out of training until its repair run, so the
    fourth diagonal push is held out and the other three are trained;
  - **runs `JOBS` (default 2) runs at once.** One SAC run leaves most of a GPU or CPU idle, so a few side
    by side get through the plan several times faster;
  - announces each run (what it trains, why), reports progress every minute (from each run's `progress.json`, which
    training writes every 1000 steps) and each outcome; training
    runs as child processes logging to `<OUT>/logs/`;
  - **pre-flight (fail fast):** text backend usable, with every operator text the runs need embedded and cached
    (with or without the name, as the runs have it; default `TEXT_BACKEND=openai`); device (`DEVICE=cuda` fails
    if unavailable); scripted policies; disk space;
  - **`RQ1_STEPS=auto`:** each rq1 run's budget is 1.5 × the sum of the steps each skill needed alone to
    reach the threshold in its learnability run (a skill that never did counts as `LEARN_STEPS`), so every
    skill gets at least what it needs alone; the rq1 runs wait for the learnability runs;
  - **order:** runs start in plan order, the most informative first: every learnability run, then every rq1
    run, then the repairs;
  - **replay buffers:** rq1 runs save theirs (about 1 GB each), and each repair starts from it (hard-linked,
    not copied, and not saved again), so the skills it was trained on keep being replayed. A run resumed
    without a buffer acts without updating for its first 5k steps (`RESUME_WARMUP_STEPS`), instead of training
    on the few transitions collected since the resume (in the 2026-10-11 repairs, resumed without buffers, the
    critic fitted those and the other skills' success dipped);
  - Ctrl+C saves and stops every run, and the same command resumes;
  - outputs: run directories per group, `logs/`, `failed.txt`, `summary.csv`, and `review.txt` at the end.
- **`scripts/review_runs.py`:** report covering runs, per-condition metrics (over seeds, with confidence intervals:
  §3), curves, TensorBoard training statistics including embedding magnitudes, added and repaired skills (their
  transfer gain, the steps from scratch in their learnability run divided by the steps after the repair, and a
  `REPAIR INVALID` flag when repair-hidden succeeds zero-shot), and learned embedding geometry.
- **`scripts/run_skill.py`:** view any skill with its scripted, random or trained policy (`--render`).
- **`scripts/watch_best_policies.py`:** for each skill, play the best trained policy in a folder of runs (the
  newest overnight folder by default), even while it is still training.
- **`scripts/check_policy_progress.py`:** is a trained policy stuck, or slowly getting to its goal? Plays each
  skill of the given runs past the training step limit (300 steps) and gives each episode a verdict (success,
  failed, progressing, stuck), with its distance to the goal, whether the block is gripped, and how much the
  policy still explores where it ends up.
- **`scripts/wrong_embedding_control.py`:** the wrong-embedding control: plays a run's policy on one skill's
  episodes given its own operator input and then another skill's (e.g. unstack-raised given unstack's). Unchanged
  success and behaviour mean the policy does not use the operator to tell the two apart. Works with every encoder,
  compositional ones included.
- **`scripts/sample_repair_scenario.sh`:** a repair scenario with PPO in a few hours (steps `check`, `train`,
  `repair`, `evaluate`). A one-hot policy trained on unstack alone is evaluated on unstack-raised with unstack's
  exact embedding, and must fail (lifting on and on is not raising). A compositional-tree policy (no operator
  name, which would carry "raised") trained on pickup, unstack and unstack-raised gets pickup-raised added and
  trained alone: its zero-shot success and steps to threshold against pickup-raised from scratch, the
  wrong-embedding control before and after, and whether each policy is stuck or slow.
- **`scripts/start_tensorboard.sh`:** `LOGDIR=<folder> ./scripts/start_tensorboard.sh`.
- **`scripts/visualize_environment.py`:** render a Fetch environment with a scripted or random policy.
- **`scripts/experimental/rectangle_grasp.py`:** a prototype of a yaw-controlled gripper picking up a
  rectangular block (a 5-D action), toward rotation skills.

### Infrastructure notes
- `--device` (auto / cpu / cuda) for the policy; it can change on resume.
- The compositional extractor composes each distinct operator once per batch (identical results; about
  2.5× faster SAC steps).
- Rendering forces GLFW's X11 build (`cam/__init__.py`), because GTK3 aborts under GNOME 50 Wayland.

## 5. Results so far

In short: no research question has a conclusive answer yet. In the 2026-10-10 run every skill was learned
alone, and every embedding condition learned all six skills when they were trained together, but the RQ1
comparison had one seed and the repair test turned out not to need new behaviour. The 2026-10-13 run, with
twice as many skills, learned almost nothing because of the gripper force penalty. Each finding below names
the rule it led to, described above.

### Overnight 2026-10-08 (PPO, pickup + putdown, unnormalized compositional embedding)
SAC learned pickup about 6× more sample-efficiently than PPO. Putdown was solved from the start, so
RQ1 was effectively single-skill and all non-compositional conditions were indistinguishable. Tree never
learned pickup because its embedding was about 100–500× larger than the other inputs, which led to the
normalization of compositional embeddings (§2).

### Overnight 2026-10-10 (SAC, 6 skills, one seed, 500k steps)
- **Learnability:** every skill learned alone within 150k steps: unstack and unstack-raised at 50k;
  pickup, pickup-raised and place-beside at 100k; stack at 150k. **Putdown succeeded untrained (step 0):
  trivial**, which is why putdown is also set up from pickup-raised, 30 cm up, where letting go is a drop.
- **Compositional learns** with normalization: tree and slots reach 100% on all six skills, on par with
  the others.
- **RQ1, inconclusive:** every condition ends at 100% on every skill. Steps to 95% differ by condition
  (e.g. stack: text-mock 250k, slots 300k, one-hot/tree 350k, random 400k, multi-hot-trainable 450k), but
  with one seed and non-monotonic curves this is within noise. Skills that only the operator tells apart
  (pickup vs pickup-raised; putdown vs stack vs place-beside) were learned by every condition, including
  one-hot. When every skill is in training, identity suffices; structure can only show up as speed, which
  needs more seeds.
- **Repair, invalid:** `unstack-raised` was solved zero-shot even when the policy could not see the
  repair (one-hot and random repair-hidden: 1.00). In that run `raised` meant 10 cm above the table, so a
  stacked block needed only 5 cm more lift, one control step at most (Fetch moves ≤ 5 cm per step). The
  scripted policies show it: unstack in 25 steps, unstack-raised in 26. A policy that keeps lifting
  satisfied it for free, and pickup-raised taught it to keep lifting. This led to `raised` as a 30–36 cm band
  that must be held, to settled success, and to pickup and unstack holding the block low
  (`(not (raised ?o))`). Repair runs also evaluated only every 50k steps, so steps to threshold could not
  separate conditions; this led to the dense evaluation of added skills (§3).

### Overnight 2026-10-13 (multi-task SAC, competence sampling, 12 skills, one seed, 500k steps)
- **Nothing learned except putdown** (0.85 alone); every RQ1 condition stayed at 0 on every skill.
- **Cause: the force penalty and failure.** The penalty was linear in the net contact force, and contact in
  the simulation spikes for single steps (up to ~38 kN): a random-action pickup episode cost about −195 in
  force penalty, against +31 for the scripted success and about 0 for doing nothing, and a single step over
  500 N ended 7 of 10 random episodes. Policies learned to stay out of contact.
- **Consequence:** the penalty was capped (2 per step, since lowered to 0.2) and failure needs 5 steps over 500 N
  (§1). With these,
  random pickup episodes cost about −10 in force penalty and 2 of 10 fail on force; the scripted policies are
  unaffected.
