# Project status and fixes

State of the code, what the experiments so far show, and what to change next. Written 2026-10-11,
updated on branch `rq-fixes`. Each section says what is done and what is still open.

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

**Fixes** (A is implemented: `--composition-weight-decay`, default 0.01):

| fix | how it works | trade-off |
| --- | --- | --- |
| **A. Decoupled weight decay on the composition only** (recommended) | after each gradient update, multiply every composition parameter by (1 − lr·wd), as AdamW does | standard remedy for weights whose output is normalized; one setting (wd) to tune |
| B. Function-preserving rescaling | LayerNorm ignores the scale of its input, so periodically dividing the composition's last layer by its current output magnitude changes nothing the policy computes and resets the raw magnitude to about 1 | exact, no tuning; only the last layer is reset, and Adam's moment estimates briefly mismatch |
| C. Bounded components | `nn.Embedding(..., max_norm=1)` for the type, variable and predicate embeddings, plus spectral norm on every composition layer | raw output bounded by construction; constrains capacity, and spectral norm costs compute every step |
| D. Normalize after every stage | LayerNorm/RMSNorm after TYPED, LITERAL, AND, PRE/EFF, OPERATOR | bounds activations, but the weights can still drift; best combined with A |
| E. Norm-aware optimizer (LAMB/LARS) | step size proportional to ‖w‖ | treats the symptom; a different optimizer for one module |
| F. Penalty on the raw magnitude | a backward hook adds the gradient of λ(‖raw‖ − 1)² | direct, but hacky inside Stable-Baselines3's losses |

**File changes for A** (done):
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

Tune wd by watching `operator_embedding_magnitude_before_normalization/*`: it should settle at O(1–10)
instead of growing.

## 2. State of the code

### Pipeline
Fetch simulator (`FetchMultiBlock-v0`, 1–5 blocks), wrapped from the inside out:
1. `FetchEnvStateAnnotationWrapper`: named state, including block positions and rotations.
2. `FetchPredicateEvaluationWrapper`: the facts true in the current state.
3. `SkillEnvironment`: per episode, picks a skill, runs its setup chain with scripted policies, picks a
   random applicable grounding, and computes success and reward.
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
| `raised(b)` | held, centre at least `RAISED_HEIGHT` = **0.30** above table rest (25 cm above a stacked start) |
| `beside(b, c)` | both on the table, centres 5.5–10 cm apart (reset places blocks ≥ 10 cm apart) |
| `at-left/right/front/back(b)` | on the table, ≥ 10 cm from the centre of the reset area (1.34, 0.75) in that direction (front = toward the robot) |

Grasping skills succeed only once the block is lifted, and placing skills only once it is released.
Success is **settled**: the effects must hold for `--success-hold-steps` (default 5) consecutive steps. A
placing skill **fails** if its block is let go in the air (`dropped`: more than 2 cm above the table, not on
a block).

### Skills (`skills/blocksworld_skills.py`, PDDL in `domain/action_model_library/pddl/`)

| skill | operator | setup chain | scripted policy | shaped reward |
| --- | --- | --- | --- | --- |
| pickup(?o) | classic blocksworld + `(not (raised ?o))` (hold it low) | – | approach, descend, close, lift | staged pickup |
| putdown(?o) | classic | **pickup-raised** (starts 30 cm up) | lower, open | staged putdown (gentle, square, smoothness, jerk) |
| stack(?o ?u) | classic | pickup | carry up, align, lower, open | staged stack (+ smoothness, jerk) |
| unstack(?o ?u) | classic + `(not (raised ?o))` | pickup, stack | pickup motion | pickup's stages, lift from the block below |
| pickup-raised(?o) | pickup + `(raised ?o)` | – | pickup motion to 33 cm | pickup's stages, lift target 31 cm |
| unstack-raised(?o ?u) | unstack + `(raised ?o)` | pickup, stack | pickup motion to 33 cm | unstack's stages, lift target 26 cm above the base rest |
| place-beside(?o ?u) | putdown + ?u + `(beside ?o ?u)` | pickup | stack motion, landing next to ?u along x | stack's stages, target ring around ?u |
| push-left/right/front/back(?o) | pre `(not (at-X ?o))`, eff `(at-X ?o)` | – | behind the block, lower, push | reach behind, push progress |
| push-left-front(?o) | push-left + `(at-front ?o)` (held out) | – | two straight pushes | as push |
| push-beside(?o ?u) | pre `(not (beside ?o ?u))`, eff `(beside ?o ?u)` | – | push toward ?u | as push |

Every scripted policy succeeds with 5-step settled success: the pick/place skills 10/10 and the pushes 18–20/20
(with 3 blocks, another block sometimes lies on a straight push path), with 2 and 3 blocks.

### Operator embedding conditions (`--operator-encoder`)
- **Identity only:** one-hot (indexed by registry order, so indices don't depend on the training
  skills); random (a fixed vector per name).
- **Structure, nothing learned:** multi-hot (literal features: 192 of `--operator-embedding-dim` 256); text (a paragraph or the PDDL, embedded by OpenAI `text-embedding-3-small` or the
  offline mock, which only matches shared character snippets).
- **Learned:** `--trainable-operator-embedding` (a learnable linear layer on top); compositional
  (tree / slots / geometric, composed inside the policy and normalized to magnitude 1).
- `--operator-identity-aliases NEW=OLD` makes one-hot and random encode NEW exactly as OLD. This is the
  repair-hidden control.

### Metrics (`evaluation/`)
Evaluated every `--eval-interval` steps with `--eval-episodes` per skill, deterministic and stochastic:
- **success rate**;
- **steps / episodes to threshold:** from the skill's first evaluation, until success ≥ 0.95;
- **zero-shot:** the first evaluation of a skill;
- **final success**;
- **AUC:** the *average* success rate over the run (area under the curve divided by the steps). It stays
  low if early success is low, even when the run ends at 100% (learnability pickup: 0 / 0.4 / 1 / 1 at
  0/50k/100k/150k gives AUC 0.63). Compare AUC only between runs with the same budget.

### Scripts
- **`scripts/start_overnight.sh`:** checks the environment (ignores the shell's `PYTHONPATH`, e.g. ROS),
  runs the unit tests, writes `run_info.txt`, blocks sleep, then runs `overnight_runs.py` in the
  foreground. Use tmux for overnight runs.
- **`scripts/overnight_runs.py`:**
  - runs one at a time, printing each run's purpose and live progress; training runs as a child process
    logging to `<OUT>/logs/`;
  - pre-flight (fail fast): text backend usable, with every text embedded and cached (default
    `TEXT_BACKEND=openai`); device; scripted policies; disk space;
  - Ctrl+C saves and stops, and the same command resumes;
  - groups learnability, rq1 and repair (repair-visible / repair-hidden), with a list of repair pairs
    (`REPAIRS`, default `unstack:unstack-raised stack:place-beside push-left:push-left-front`);
  - repaired skills are evaluated every 5k steps (`--stop-eval-interval`);
  - conditions by default: one-hot, random, multi-hot-trainable, text, compositional tree / slots /
    geometric.
- **`scripts/review_runs.py`:** report covering runs, per-condition metrics, curves, TensorBoard training
  statistics including embedding magnitudes, added and repaired skills (with an **invalid repair** flag
  when repair-hidden succeeds zero-shot), and learned embedding geometry.
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

## 4. Changes since that run
On `repair-experiments` (merged in PR #7):
- `raised` raised to 20 cm; a sequential overnight runner with pre-flight checks (OpenAI text by
  default) and resume; compositional-geometric in the default conditions; repair variants renamed
  repair-visible / repair-hidden; an invalid-repair flag in the review report; `--device`.

On `rq-fixes`:
- **Settled success** (`--success-hold-steps 5`).
- **Hold low unless asked:** pickup and unstack get `(not (raised ?o))`, and `raised` is now 30 cm.
- **Non-trivial putdown:** set up by pickup-raised; a drop is a failure; smoothness and jerk penalties in
  putdown and stack.
- **Composition weight decay** (`--composition-weight-decay 0.01`).
- **Repair resolution:** stop skills are evaluated every 5k steps, with no redundant evaluation after a
  stop.
- **Global regions and push skills,** with `push-beside`; `--operator-embedding-dim` 256.
- **Runner:** a list of repair pairs; the pushes are in the default training skills.

## 5. Known issues and planned fixes
Done on `rq-fixes`: putdown non-triviality, settled success, raised reached by overshoot, repair
resolution, held-out operators with literals seen in training, several repair pairs, and the redundant
evaluation. Still open:

1. **Sequential runtime.** The default plan is 47 runs: 13 learnability; 7 RQ1 on 10 skills; 27 repair
   (3 pairs × 7 conditions, plus 2 repair-hidden per pair). At single-process CPU speeds that's about
   **83 hours**. A GPU may help, unmeasured. Options:
   - fewer conditions or steps per night (`CONDITIONS`, `*_STEPS`);
   - groups split across nights (`RUN_GROUPS`);
   - parallel jobs on many-core CPUs (the parallel bash runner is in git history,
     `d6c7c0d3:scripts/overnight_runs.sh`).
2. **The RQ1 budget.** 500k steps for 10 skills is about 50k per skill. Learnability shows single skills
   need 50–150k, so expect partial learning. Consider `RQ1_STEPS=1000000` once runtime allows.
3. **One seed.** Use 3+ seeds before ranking conditions.
4. **`stack` → `place-beside` is a larger edit** than the other repairs: a parameter's role, the
   preconditions and several effects change. Treat it as a new-skill test more than a one-literal repair.
5. **Resumed SAC runs start with an empty replay buffer** (it isn't saved by default; about 1 GB per run).
   Repair runs start fresh, the same for every condition.
6. **Old runs don't resume:** the predicate set changed again (regions), so multi-hot features,
   compositional ids and the embedding size (256) differ.
7. **Run logs are large** (40–80 MB each, mostly progress-bar redraws). The runner reads progress from
   them.

## 6. Future work
- **RQ2 (new skills):** the protocol from `docs/research_focus.md`: add skills one at a time to a
  growing training set, measure steps to threshold for each new skill, and shuffle the order across
  seeds. The repair mechanism (resume with added skills) already supports this.
- **RQ4 (hypernetworks):** first FiLM conditioning (the embedding scales and shifts the state features),
  then a real hypernetwork: a custom SB3 SAC actor whose action head weights come from the operator
  embedding (low-rank: W(e) = U·diag(h(e))·V). The learnability runs are the "individually trained
  policies" baseline.
- **RQ5 (does the embedding encode structure?):** linear probes from each learned embedding to its
  literals, with the multi-hot vector as the target.
- **RQ6 (behavioural sensitivity):** remove one literal at a time and compare behaviour.
- **Statistics:** 3–5 seeds per condition, mean ± std or bootstrap intervals in `review_runs.py`.

## 7. How to run
```bash
./scripts/start_overnight.sh                     # inside tmux; plan, checks, then each run explained
DRY_RUN=1 ./scripts/start_overnight.sh           # the plan and its time estimate only
RUN_GROUPS=learnability ./scripts/start_overnight.sh
CONDITIONS="one-hot compositional-tree" SEEDS="0 1 2" ./scripts/start_overnight.sh
./scripts/review_runs.py outputs/overnight_<date>   # the report, at any time
LOGDIR=outputs/overnight_<date> ./scripts/start_tensorboard.sh
```
