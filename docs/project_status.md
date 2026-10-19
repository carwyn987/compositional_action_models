# Project status

State of the code and what the experiments so far show. What to do next is in `docs/next_steps.md`.

## 1. Compositional embedding magnitude growth

**Problem.** The compositional extractor (`src/cam/representations/compositional/extractor.py`)
normalizes the composed embedding to zero mean and magnitude (L2 norm) 1 before the policy sees it. Before
that normalization, the architecture's raw output has no fixed scale:

- **At initialization:** about 50–150 for tree and 20–30 for slots and geometric (measured).
- **During training:** it grows further. In the 2026-10-10 overnight run, slots went from about 2.5 to
  800–2800 over 500k SAC steps (`operator_embedding_magnitude_before_normalization/*` in TensorBoard).

The policy never sees this scale. Before normalization was added, the tree's huge embedding swamped the
MLP's other inputs, and tree never learned pickup.

**Why it still matters.** The normalized output doesn't depend on the composition's scale, so nothing in
the loss pushes the scale back down, and the weights drift upward. Adam keeps each step's size roughly
constant, so a step changes the normalized output less as the weights grow. The composition learns more
slowly late in training. An earlier note said "1000× slower": that overstates it. The slowdown is set by
each layer's weight growth, not by the full output growth, and hasn't been measured.

**Fixes** (A is implemented: `--composition-weight-decay`, default 0.01, compositional runs only):

| fix | how it works | trade-off |
| --- | --- | --- |
| **A. Decoupled weight decay on the composition only** (recommended) | after each gradient update, multiply every composition parameter by (1 − lr·wd), as AdamW does | standard remedy for weights whose output is normalized; one setting (wd) to tune |
| B. Function-preserving rescaling | LayerNorm ignores the scale of its input, so periodically dividing the composition's last layer by its current output magnitude changes nothing the policy computes and resets the raw magnitude to about 1 | exact, no tuning; only the last layer is reset, and Adam's moment estimates briefly mismatch |
| C. Bounded components | `nn.Embedding(..., max_norm=1)` for the type, variable and predicate embeddings, plus spectral norm on every composition layer | raw output bounded by construction; constrains capacity, and spectral norm costs compute every step |
| D. Normalize after every stage | LayerNorm/RMSNorm after TYPED, LITERAL, AND, PRE/EFF, OPERATOR | bounds activations, but the weights can still drift; best combined with A |
| E. Norm-aware optimizer (LAMB/LARS) | step size proportional to ‖w‖ | treats the symptom; a different optimizer for one module |
| F. Penalty on the raw magnitude | a backward hook adds the gradient of λ(‖raw‖ − 1)² | direct, but hacky inside Stable-Baselines3's losses |

**A, as implemented** (`CompositionWeightDecayCallback`):
- `src/cam/training/stable_baselines3_trainer.py`: add `CompositionWeightDecayCallback(weight_decay)`.
  On each step it reads how many gradient updates happened since the last call (`model._n_updates`) and
  multiplies every parameter under each features extractor's `.composition` by (1 − lr·wd)^updates.
  That covers the actor and critic extractors for SAC, and the shared one for PPO. It keeps no state, so
  resuming works.
- `main.py`: a `--composition-weight-decay` option (start around 0.01), not locked on resume; `train()`
  adds the callback for compositional runs.
- `tests/cam/training/`: the callback shrinks composition parameters and only those; a short run keeps
  `operator_embedding_magnitude_before_normalization` bounded.
- `scripts/overnight_runs.py`: pass the option to compositional runs.

Weight decay has no task-specific learning direction: it shrinks every parameter toward zero, and the
learning signal has to keep re-growing what it needs. Too strong, and the composition underfits and
operators blur together. It also shrinks the embeddings of rarely seen predicates, which get few updates to
recover. Tune wd by watching `operator_embedding_magnitude_before_normalization/*`: it should settle at
O(1–10) instead of growing.

## 2. State of the code

### Pipeline
Fetch simulator (`FetchMultiBlock-v0`, 1–5 blocks), wrapped from the inside out:
1. `FetchEnvStateAnnotationWrapper`: named state, including block positions and rotations.
2. `FetchPredicateEvaluationWrapper`: the facts true in the current state.
3. `SkillEnvironment`: per episode, picks a skill (`--skill-sampling`, below), runs one of its setup chains with scripted policies,
   picks a random applicable grounding, and computes success, failure and reward.
4. `TimeLimit`: outside `SkillEnvironment`, so setup steps don't count toward the episode's limit.
5. `PolicyObservationWrapper`: builds `{observation, operator_embedding, grounding}`.

**Skill sampling** (`--skill-sampling`): **round-robin** (the skills take turns) or **competence**: each skill
keeps half its even share of the episodes, and the rest goes to skills in proportion to their recent failure
rate (over their last 20 training episodes; an episode cut off by the time limit is a failure), so skills
that are learned give their episodes to skills that are not. The choice is a smooth weighted round-robin, so
the shares hold without sampling noise. The overnight runner uses competence for every condition. A skill
that cannot be learned keeps a large share, so gate such skills out of the training set.

Design rules the code keeps:
- `cam.domain` (symbols, action models, the PDDL parser and writer) imports neither Gymnasium, NumPy nor
  torch.
- Facts are closed-world and use the action models' predicate names and arities. They follow the PDDL
  meaning, not physical intuition: a held block is not `clear`.
- Predicates are functions of a `FetchState` (named fields), so they are unit-tested on hand-built states
  without the simulator.
- Setup acts through the normal `step` API, never by setting the simulator state, so every set-up state
  is physically reachable.

Training is Stable-Baselines3 SAC (default), multi-task SAC (`--algorithm mtsac`, the overnight runner's
default) or PPO, with features extractors for the trainable and compositional embeddings. Multi-task SAC
keeps a separate entropy temperature per skill (`MultiTaskSAC`, as on Meta-World): with one temperature,
the entropy target holds only on average over the skills, so learned skills and skills still being explored
pull it in opposite directions. Each distinct operator embedding gets its own temperature, logged as
`train/ent_coef/<skill>`; the policy's input is unchanged. See `docs/operator_embeddings.md` for every embedding type, with worked
examples.

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
| `raised(b)` | held, centre **30–36 cm** above table rest (a band: it must be stopped there, not just passed) |
| `beside(b, c)` | both on the table, centres 5.5–10 cm apart (reset places blocks ≥ 10 cm apart) |
| `at-left/right/front/back(b)` | on the table, **10–18 cm** from the centre of the reset area (1.34, 0.75) in that direction (front = toward the robot, left = +y) |

**Success is settled and respects the frame:** the effects must hold for `--success-hold-steps` (default 5)
consecutive steps, and every fact the action does not mention must keep its value from the episode's start
(the STRIPS frame assumption, checked over `on-table`, `on`, `clear`, `holding`, `gripper-empty`; the
geometric predicates change as side effects and are left out). So unstack(?o ?u) must leave `(on-table ?u)`
true: lifting both blocks is not an unstack.

**Failure** ends an episode unsuccessfully, with a penalty of 10 under shaped rewards, and takes precedence
over success. It is the same for every skill (`check_fetch_failure`): an **extreme force** on the gripper (net
contact force above `MAX_GRIPPER_FORCE`, 500 N, for `FORCE_FAILURE_STEPS`, 5, steps in a row: pressing on a
block or the table), the moved block (first argument) was **dropped** (let go
more than `DROP_HEIGHT`, 2 cm, above its support), or another block was **disturbed** (moved more than
`DISTURB_DISTANCE`, 3 cm, from where it started: knocked or pushed aside).

**Motion penalties** are added to every skill's shaped reward (`FetchMotionReward`): action changes
(smoothness), the moved block's acceleration, any movement of the other blocks, and the **gripper force**
above `SAFE_GRIPPER_FORCE` (100 N), counted up to 500 N: at most 2 per step, so pressing until it fails
costs at most twice the failure, and a shorter bump less than one. The gripper force is the net contact force
on the gripper and fingers, read from MuJoCo (`FetchState.gripper_force`, with the last 10 steps' magnitudes
in `gripper_force_history`): a grasp's squeeze cancels out, so it measures pressing on a block or the table,
pushing, ramming, and a carried block's weight (~20 N). Lifting and placing peak around 55 N, steady pushing
around 65 N; pressing into the table stays around 55 kN. First contact spikes for a single step (ramming a
block with closed fingers: ~900 N once; random exploration: up to ~38 kN on about 1% of steps), and random
exploration bumps the table for a few steps at a time, which is why the penalty is capped and failure needs 5
steps. This matters for moving to real robots, whose grippers such forces would damage. Grasping skills therefore
succeed only once the block is lifted and held, and placing skills only once it is released, gently, without
disturbing anything else.

### Skills (`skills/blocksworld_skills.py`, PDDL in `domain/action_model_library/pddl/`)

| skill | operator | setup | scripted policy | shaped reward |
| --- | --- | --- | --- | --- |
| pickup(?o) | classic blocksworld + `(not (raised ?o))`: hold it low | – | approach, descend, close, lift | staged pickup |
| putdown(?o) | classic | pickup **or** pickup-raised (random per episode) | lower, open | staged putdown (gentle descent, square) |
| stack(?o ?u) | classic | pickup | carry up, align, lower, open | staged stack |
| unstack(?o ?u) | classic + `(not (raised ?o))` | pickup, stack | pickup motion | pickup's stages, lift from the block below |
| pickup-raised(?o) | pickup + `(raised ?o)` | – | pickup motion to 33 cm | pickup's stages, lift target 33 cm |
| unstack-raised(?o ?u) | unstack + `(raised ?o)` | pickup, stack | pickup motion to 33 cm | unstack's stages, lift target 28 cm above the base rest |
| place-beside(?o ?u) | putdown + ?u + `(beside ?o ?u)` | pickup | stack motion, landing next to ?u along x | stack's stages, target ring around ?u |
| push-left/right/front/back(?o) | pre `(not (at-X ?o))`, eff `(at-X ?o)` | – | rise, behind the block, lower, push toward the band's middle | reach behind, push progress |
| push-right-front, push-left-back, push-right-back, push-left-front(?o) | both regions | – | two straight pushes | as push |
| push-beside(?o ?u) | pre `(not (beside ?o ?u))`, eff `(beside ?o ?u)` | – | push toward ?u | as push |

In the default overnight plan (`scripts/overnight_runs.py`), `push-left-front` is held out and the other
three diagonals are trained. Every scripted policy succeeds with 5-step settled success (2 and 3 blocks):
pick/place skills 10/10, pushes 18–20/20 (with 3 blocks another block can lie on a straight push path).

### Operator embedding conditions (`--operator-encoder`, `--operator-embedding-dim 256`)
- **Identity only:** one-hot (indexed by registry order, so indices don't depend on the training
  skills); random (a fixed vector per name).
- **Structure, nothing learned:** multi-hot (192 literal features); text (a paragraph or the PDDL,
  embedded by OpenAI `text-embedding-3-small` or the offline mock, which only matches shared character
  snippets).
- **Learned:** `--trainable-operator-embedding` (a learnable linear layer on top); compositional (tree /
  slots / geometric, composed inside the policy, normalized to magnitude 1, with composition weight decay).
- `--operator-identity-aliases NEW=OLD` makes one-hot and random encode NEW exactly as OLD. This is the
  repair-hidden control.

### Metrics (`evaluation/`)
Evaluated every `--eval-interval` steps (50k) with `--eval-episodes` (20) per skill, deterministic and
stochastic. A skill added on resume (a repaired or new operator) is also evaluated alone 1k, 2k, 4k, …
steps after it was added, doubling until `--eval-interval` takes over: precise steps to threshold early,
where conditions differ most, at little cost (`ADDED_SKILL_FIRST_EVALUATION` in `metrics_callback.py`).
- **success rate**;
- **steps / episodes to threshold:** from the skill's first evaluation, until success ≥ 0.95, counted in all
  training steps and in the skill's own (**on skill**, `skill_steps_to_threshold`). Skills share a run's
  steps, so the skill's own steps are what compare across skills, across ways of sampling skills, and with
  learning the skill alone;
- **zero-shot:** the first evaluation of a skill;
- **final success**;
- **AUC:** the *average* success rate over the run (area under the curve divided by the steps). It stays
  low if early success is low, even when the run ends at 100%. Compare AUC only between runs with the same
  budget.
- **retention** (after a skill is added): for every other skill, its level before (the mean of its last 2
  evaluations), its lowest and last success after, and when it recovered (back within 0.1 of its level on 2
  evaluations in a row, counting only evaluations after the addition): how much the new skill made the
  policy forget (`"retention"` in `metrics.json`).

**Repair modes** (`--repair-mode`, resuming with the repaired skill in `--stop-skills`):
- **isolated:** only the repaired skill trains; the others are evaluated, not trained. Stops once it is at
  the threshold on 2 evaluations in a row. Its steps compare directly with learning the skill from scratch.
- **retain:** every skill trains, each episode's skill chosen as in the run it continues. Stops once the
  repaired skill is at the threshold and every other skill has recovered: the cost of a repair without
  forgetting.

### Scripts
- **`scripts/start_overnight.sh`:** checks the environment (ignores the shell's `PYTHONPATH`, e.g. ROS),
  runs the unit tests, writes `run_info.txt`, blocks sleep, then runs `overnight_runs.py`.
- **`scripts/overnight_runs.py`:**
  - **runs `JOBS` (default 2) runs at once.** One SAC run leaves most of a GPU or CPU idle, so a few side
    by side get through the plan several times faster;
  - announces each run (what it trains, why), reports progress every minute and each outcome; training
    runs as child processes logging to `<OUT>/logs/`;
  - **pre-flight (fail fast):** text backend usable, with every text embedded and cached (default
    `TEXT_BACKEND=openai`); device (`DEVICE=cuda` fails if unavailable); scripted policies; disk space;
  - **`RQ1_STEPS=auto`:** each rq1 run's budget is 1.5 × the sum of the steps each skill needed alone to
    reach the threshold in its learnability run (a skill that never did counts as `LEARN_STEPS`), so every
    skill gets at least what it needs alone; the rq1 runs wait for the learnability runs;
  - **order:** runs start in plan order, the most informative first: every learnability run, then every rq1
    run, then the repairs;
  - **replay buffers:** rq1 runs save theirs (about 1 GB each), and each repair starts from it (hard-linked,
    not copied, and not saved again), so the skills it was trained on keep being replayed. A run resumed
    without a buffer acts without updating for its first 5k steps (`RESUME_WARMUP_STEPS`), instead of training
    on the few transitions collected since the resume (in the 2026-10-11 repairs the critic fitted those, and
    the other skills' success dipped);
  - Ctrl+C saves and stops every run, and the same command resumes;
  - **groups:** learnability (each skill alone), rq1 (one shared policy per condition and seed), repair
    (each `FROM:TO` in `REPAIRS`, in each of `REPAIR_MODES`, repair-visible and repair-hidden; every skill
    evaluated every `REPAIR_EVAL_INTERVAL` (10k) steps).
- **`scripts/review_runs.py`:** report covering runs, per-condition metrics, curves, TensorBoard training
  statistics including embedding magnitudes, added and repaired skills (with an **invalid repair** flag when
  repair-hidden succeeds zero-shot), and learned embedding geometry.
- **`scripts/run_skill.py`:** view any skill with its scripted, random or trained policy (`--render`).
- **`scripts/watch_best_policies.py`:** for each skill, play the best trained policy in a folder of runs (the
  newest overnight folder by default), even while it is still training.
- **`scripts/check_policy_progress.py`:** is a trained policy stuck, or slowly getting to its goal? Plays each
  skill of the given runs past the training step limit (300 steps) and gives each episode a verdict (success,
  failed, progressing, stuck), with its distance to the goal, whether the block is gripped, and how much the
  policy still explores where it ends up.
- **`scripts/start_tensorboard.sh`:** `LOGDIR=<folder> ./scripts/start_tensorboard.sh`.

### Infrastructure notes
- `--device` (auto / cpu / cuda) for the policy; it can change on resume.
- The compositional extractor composes each distinct operator once per batch (identical results; about
  2.5× faster SAC steps).
- Rendering forces GLFW's X11 build (`cam/__init__.py`), because GTK3 aborts under GNOME 50 Wayland.
- Operator embedding magnitudes are logged every 1000 steps (`operator_embedding_magnitude/*`).

## 3. Results so far

### Overnight 2026-10-08 (PPO, pickup + putdown, before normalization)
SAC learned pickup about 6× more sample-efficiently than PPO. Putdown was solved from the start, so
RQ1 was effectively single-skill and all non-compositional conditions were indistinguishable. Tree never
learned pickup because its embedding was about 100–500× larger than the other inputs; this led to the
normalization.

### Overnight 2026-10-10 (SAC, 6 skills, one seed, 500k steps)
- **Learnability:** every skill learned alone within 150k steps: unstack and unstack-raised at 50k;
  pickup, pickup-raised and place-beside at 100k; stack at 150k. **Putdown succeeds untrained (step 0):
  trivial.**
- **Compositional now learns:** tree and slots reach 100% on all six skills, on par with the others.
- **RQ1, inconclusive:** every condition ends at 100% on every skill. Steps to 95% differ by condition
  (e.g. stack: text-mock 250k, slots 300k, one-hot/tree 350k, random 400k, multi-hot-trainable 450k), but
  with one seed and non-monotonic curves this is within noise. Skills that only the operator tells apart
  (pickup vs pickup-raised; putdown vs stack vs place-beside) were learned by every condition, including
  one-hot. When every skill is in training, identity suffices; structure can only show up as speed, which
  needs more seeds.
- **Repair, invalid:** `unstack-raised` was solved zero-shot even when the policy could not see the
  repair (one-hot and random repair-hidden: 1.00). With `raised` at 10 cm, a stacked block needed only
  5 cm more lift, one control step at most (Fetch moves ≤ 5 cm per step). The scripted policies show it:
  unstack in 25 steps, unstack-raised in 26. A policy that keeps lifting satisfies it for free, and
  pickup-raised teaches it to keep lifting. Repair runs also evaluated only every 50k steps, so
  steps-to-threshold could not separate conditions.

### Overnight 2026-10-13 (mtsac, competence sampling, 12 skills, one seed, 500k steps)
- **Nothing learned except putdown** (0.85 alone); every RQ1 condition stayed at 0 on every skill.
- **Cause: the force penalty and failure.** The penalty was linear in the net contact force, and contact in
  the simulation spikes for single steps (up to ~38 kN): a random-action pickup episode cost about −195 in
  force penalty, against +31 for the scripted success and about 0 for doing nothing, and a single step over
  500 N ended 7 of 10 random episodes. Policies learned to stay out of contact.
- **Change:** the penalty is capped (2 per step) and failure needs 5 steps over 500 N. Random pickup episodes
  then cost about −10 in force penalty and 2 of 10 fail on force; the scripted policies are unchanged.

## 4. Changes made after the 2026-10-10 run
Each addresses something that run showed:
- **Settled success,** so skills must stop in their goal state.
- **Hold low unless asked:** pickup and unstack get `(not (raised ?o))`.
- **`raised` as a 30–36 cm band:** 25 cm of lift from a stacked start, and it must be stopped and held.
- **Non-trivial putdown:** sometimes set up 30 cm up; a drop is a failure.
- **The frame:** success requires unmentioned facts to stay unchanged (unstack-raised had learned to lift both
  blocks).
- **Gentle, careful motion for every skill:** acceleration and smoothness penalties, and knocking another
  block aside is a failure with a penalty (pushes had learned to throw blocks around).
- **Gripper force limits:** net contact force on the gripper penalized above 100 N and a failure above
  500 N (needed for real robots). The scripted push moves at 0.25 of full speed while pushing, which keeps
  its contact peaks under 400 N; its diagonals need more than 100 steps, so the pre-flight check gives
  scripted policies 150.
- **Failure precedence:** a failure is never counted as success.
- **Composition weight decay.**
- **Repaired / added skills evaluated densely, then sparsely** (1k, 2k, 4k, … after being added).
- **Global regions** (10–18 cm bands) with push skills, the four diagonals (one held out) and
  push-beside, so every literal of every held-out operator is seen in training.
- **Runner:** a list of repair pairs; several runs at once.

## 5. Next steps
Open issues, feature additions and the direction for each research question are in `docs/next_steps.md`, the
project's single backlog.

## 6. How to run
```bash
./scripts/start_overnight.sh                       # plan, checks, then 2 runs at a time, each explained
DEVICE=cuda ./scripts/start_overnight.sh           # on a GPU
DRY_RUN=1 ./scripts/start_overnight.sh             # the plan and its time estimate only
RUN_GROUPS=learnability JOBS=6 ./scripts/start_overnight.sh
CONDITIONS="one-hot compositional-tree" SEEDS="0 1 2" ./scripts/start_overnight.sh
./scripts/review_runs.py outputs/overnight_<date>  # the report, at any time
LOGDIR=outputs/overnight_<date> ./scripts/start_tensorboard.sh
./scripts/watch_best_policies.py                   # the best trained policy for each skill, rendered
./scripts/run_skill.py --policy model --run-directory <run> --skill <skill> --render   # one run and skill
```
