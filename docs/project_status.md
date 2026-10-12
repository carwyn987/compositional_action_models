# Project status

State of the code, what the experiments so far show, what is still open, and what to change next.

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
3. `SkillEnvironment`: per episode, picks a skill, runs one of its setup chains with scripted policies,
   picks a random applicable grounding, and computes success, failure and reward.
4. `TimeLimit`.
5. `PolicyObservationWrapper`: builds `{observation, operator_embedding, grounding}`.

Training is Stable-Baselines3 SAC (default) or PPO, with features extractors for the trainable and
compositional embeddings. See `docs/operator_embeddings.md` for every embedding type, with worked
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

**Success is settled:** the effects must hold for `--success-hold-steps` (default 5) consecutive steps.
**Failure conditions** end an episode unsuccessfully and take precedence over success: a step where one
holds never counts toward success. The placing skills (putdown, stack, place-beside) fail if their block is
**dropped**, i.e. let go more than `DROP_HEIGHT` (2 cm) above its support, the table or the block below.
Grasping skills therefore succeed only once the block is lifted (and held), and placing skills only once it
is released.

### Skills (`skills/blocksworld_skills.py`, PDDL in `domain/action_model_library/pddl/`)

| skill | operator | setup | scripted policy | shaped reward |
| --- | --- | --- | --- | --- |
| pickup(?o) | classic blocksworld + `(not (raised ?o))`: hold it low | – | approach, descend, close, lift | staged pickup |
| putdown(?o) | classic | pickup **or** pickup-raised (random per episode) | lower, open | staged putdown (gentle, square, smoothness, jerk) |
| stack(?o ?u) | classic | pickup | carry up, align, lower, open | staged stack (+ smoothness, jerk) |
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
- **steps / episodes to threshold:** from the skill's first evaluation, until success ≥ 0.95;
- **zero-shot:** the first evaluation of a skill;
- **final success**;
- **AUC:** the *average* success rate over the run (area under the curve divided by the steps). It stays
  low if early success is low, even when the run ends at 100%. Compare AUC only between runs with the same
  budget.

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
  - Ctrl+C saves and stops every run, and the same command resumes;
  - **groups:** learnability (each skill alone), rq1 (one shared policy per condition and seed), repair
    (each `FROM:TO` in `REPAIRS`, repair-visible and repair-hidden).
- **`scripts/review_runs.py`:** report covering runs, per-condition metrics, curves, TensorBoard training
  statistics including embedding magnitudes, added and repaired skills (with an **invalid repair** flag when
  repair-hidden succeeds zero-shot), and learned embedding geometry.
- **`scripts/run_skill.py`:** view any skill with its scripted, random or trained policy (`--render`).
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

## 4. Changes made after the 2026-10-10 run
Each addresses something that run showed:
- **Settled success,** so skills must stop in their goal state.
- **Hold low unless asked:** pickup and unstack get `(not (raised ?o))`.
- **`raised` as a 30–36 cm band:** 25 cm of lift from a stacked start, and it must be stopped and held.
- **Non-trivial putdown:** sometimes set up 30 cm up; a drop is a failure; smoothness and jerk penalties.
- **Failure precedence:** a failure is never counted as success.
- **Composition weight decay.**
- **Repaired / added skills evaluated densely, then sparsely** (1k, 2k, 4k, … after being added).
- **Global regions** (10–18 cm bands) with push skills, the four diagonals (one held out) and
  push-beside, so every literal of every held-out operator is seen in training.
- **Runner:** a list of repair pairs; several runs at once.

## 5. Open issues and proposals
1. **Runtime.** The default plan is 50 runs: 13 learnability; 7 RQ1 on 13 skills; 27 repair (3 pairs × 7
   conditions, plus 2 repair-hidden per pair). With 2 jobs the first estimate is about two days at typical CPU
   speeds; the runner refines it after the first runs. Trim with `RUN_GROUPS`, `CONDITIONS` or `*_STEPS`,
   or raise `JOBS`.
2. **RQ1 budget.** 500k steps for 13 skills is about 40k per skill, and single skills need 50–150k, so
   expect partial learning. Consider `RQ1_STEPS=1000000` once runtime allows.
3. **One seed.** Use 3+ seeds before ranking conditions.
4. **`stack` → `place-beside` is a larger edit** than the other repairs (a parameter's role, the
   preconditions and several effects change). Read it as a new-skill test more than a one-literal repair.
5. **Resumed SAC runs start with an empty replay buffer** (it isn't saved by default; about 1 GB per run),
   the same for every condition.
6. **Old runs don't resume:** the predicate set and the embedding size (256) changed.
7. **Run logs are large** (40–80 MB each, mostly progress-bar redraws). The runner reads progress from
   them.

## 6. Future work
- **Rotation skills.** For example `rotate-90(?o)` and `rotate-180(?o)`, with a held-out in-between angle
  (`rotate-45`) or a chain (rotate, then place offset) to test generalization. This needs two things the
  setup lacks:
  - **a way to rotate:** Fetch's action is (dx, dy, dz, gripper) with a fixed downward wrist, so the robot
    can only turn a block by pushing it off-centre. That calls for an environment variant with wrist yaw
    control (a 5-D action);
  - **orientation predicates:** global yaw bands from the block's orientation (e.g. `yaw-0`, `yaw-45`,
    `yaw-90`), like the regions. A cube looks the same every 90°, but the simulator tracks its body yaw, so
    the predicates can use it.
- **A drop skill** (`drop(?o)`: holding → on-table, released from height). Its effects would match
  putdown's, so it can only be told apart from putdown by its name, plus a literal such as a `(dropped ?o)`
  marker. Use: a control for operators that are structurally identical but behave differently (where
  structure-based embeddings *should* fail), and a contrast that makes "lower first" depend on the operator.
- **RQ2 (new skills):** the protocol from `docs/research_focus.md`: add skills one at a time to a
  growing training set, measure steps to threshold for each new skill, and shuffle the order across seeds.
  Resume-with-added-skills already supports it.
- **RQ4 (hypernetworks):** first FiLM conditioning (the embedding scales and shifts the state features),
  then a real hypernetwork: a custom SB3 SAC actor whose action head weights come from the operator
  embedding (low-rank: W(e) = U·diag(h(e))·V). The learnability runs are the "individually trained
  policies" baseline.
- **RQ5 (does the embedding encode structure?):** linear probes from each learned embedding to its
  literals (multi-hot as the target). A 2-D PCA plot of embeddings moving during training would help; one
  existed in the original implementation (`plot_embeddings.py`, git commit 59054395).
- **RQ6 (behavioural sensitivity):** remove one literal at a time and compare behaviour.
- **From the original implementation** (git commit 59054395): behaviour-cloning warm starts from the
  scripted policies (`fetch_blockworld/teacher.py`), keyboard teleoperation with live facts
  (`experimental/manual_fetch.py`), and a penalty on rotating the block in the push reward.
- **Statistics:** 3–5 seeds per condition, mean ± std or bootstrap intervals in `review_runs.py`.

## 7. How to run
```bash
./scripts/start_overnight.sh                       # plan, checks, then 2 runs at a time, each explained
DEVICE=cuda ./scripts/start_overnight.sh           # on a GPU
DRY_RUN=1 ./scripts/start_overnight.sh             # the plan and its time estimate only
RUN_GROUPS=learnability JOBS=6 ./scripts/start_overnight.sh
CONDITIONS="one-hot compositional-tree" SEEDS="0 1 2" ./scripts/start_overnight.sh
./scripts/review_runs.py outputs/overnight_<date>  # the report, at any time
LOGDIR=outputs/overnight_<date> ./scripts/start_tensorboard.sh
./scripts/run_skill.py --policy model --run-directory <run> --skill <skill> --render   # watch a trained skill
```
