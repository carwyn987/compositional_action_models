# Next steps

The single backlog for this project: known issues, feature additions and the direction for each research
question. It is written so a new session can start from it. Read it together with:

- `docs/research_focus.md`: the research questions (RQ1–RQ6).
- `docs/project_status.md`: what the code does today (predicates, skills, embeddings, metrics, scripts) and
  what past overnight runs showed.

When an item is done, delete it here and describe the result in `project_status.md`.

## 0. Starting a new session

Give the session the latest overnight results. From the repository on the training machine:

```bash
OUT=$(ls -d outputs/overnight_*/ | tail -1)
cat $OUT/run_info.txt                       # the commit and settings the run used
tail -3 $OUT/overnight.log                  # finished? ("all runs done" or "All done")
cat $OUT/failed.txt 2>/dev/null             # runs that crashed
./scripts/review_runs.py $OUT --no-embeddings | sed -n '/^1\./,/^3\./p;/^5\./,/^6\./p'   # runs, conditions, repairs
```

Then ask it to read this file, `project_status.md` and `research_focus.md`, and to propose what to do next
using §2 below.

**Check the run's commit first.** Runs started before the frame check, the motion and force penalties and the
disturbance failure were merged (PR #9, commit `7077802f`) were trained without those constraints. Their
successes may include exploits such as lifting both blocks, so don't compare them with newer runs.

## 1. Working agreements

How changes are made here (keep doing this):

- **Branch and PR for every change**, one commit per cohesive feature or fix, with a descriptive commit
  message, so each commit reviews on its own; the PR description lists the commits. Touch as few files as
  possible and keep additions minimal.
- **Tests for everything; functional tests for behaviour.** A test of a safeguard includes a control
  showing that without it the problem happens (otherwise the test may pass for the wrong reason).
- **Scripted policies are the ground truth for a skill's setup, predicates and reward.** Every skill's
  scripted policy must succeed (see `scripts/run_skill.py`) before anything trains on it.
- **Code comments and docs describe the code as it is**: no history ("now", "previously", "we changed"),
  no branch names. Docs describe master.
- **Which skills are trained or held out is decided by the run plan** (`scripts/overnight_runs.py`),
  not by skill definitions.
- **The OpenAI key** lives in `~/.secrets/openai_api_key` (or `OPENAI_API_KEY`); never print, log or commit
  it. Check diffs for key-like strings before pushing.
- **The training machine** has a GPU and a ROS setup sourced in its shell. `scripts/start_overnight.sh`
  ignores the shell's `PYTHONPATH` and outside pytest plugins for that reason; run it inside tmux.

## 2. Reading the overnight results: what to do next

| what the report shows | likely cause | next step |
| --- | --- | --- |
| A skill doesn't reach 95% in its **learnability** run | reward stages, step limit, or the constraints (failure rate) | Look at its `failure` reasons in the logs and `watch_best_policies.py`. For the pushes, try `--max-steps-per-episode 150` (the gentle scripted diagonal needs ~105–120 steps). If failures dominate, revisit the thresholds (§3.2). |
| Many episodes end in **failure** (`disturbed`, `excessive force`, `dropped`) | constraints too strict for exploration, or a real behaviour problem | Watch the policy. Disturbance is 3 cm, force 500 N (penalty above 100 N); relax only if the behaviour is acceptable. |
| **RQ1** conditions are indistinguishable | one seed, or too small a budget (13 skills in 500k steps is ~40k each) | 3+ seeds, `RQ1_STEPS=1000000`; then add confidence intervals (§4.1). |
| A **compositional** condition lags | composition scale (watch `operator_embedding_magnitude_before_normalization/*`) or capacity | Tune `--composition-weight-decay`; compare tree / slots / geometric. |
| **Repair: repair-hidden succeeds zero-shot** (`REPAIR INVALID` in section 5) | the original behaviour already satisfies the repaired operator | Redesign that repair pair so the old behaviour fails it. |
| **Repair: repair-visible zero-shot is ~0 everywhere** | the held-out operator is too far from training, or the policy ignores the embedding | Check with a wrong-embedding control (§5, RQ1). Try the easier pair (`push-left` → `push-left-front`) first. |
| Repair steps-to-threshold vary but look noisy | one seed | more seeds. |

## 3. Issues to fix

In rough priority order.

1. **Run directory names grow with the skill list.** With the 13 default skills a name is 185 characters,
   and the filesystem limit is 255. A few more skills, or longer skill names, will break run creation.
   Shorten them, e.g. `13skills-<short hash>` once the list is long.
2. **Thresholds are first guesses.** `DISTURB_DISTANCE` (3 cm), `SAFE_GRIPPER_FORCE` (100 N),
   `MAX_GRIPPER_FORCE` (500 N), `DROP_HEIGHT` (2 cm) and the motion penalty weights were calibrated on the
   scripted policies. Set the force limits from the real gripper's rating if a real robot is planned. The
   gripper is position-controlled in simulation, so first contact spikes for one step; failure therefore
   needs 5 steps over 500 N, and a single hard impact (e.g. ramming a block) is only penalized. Telling a
   ram from a first contact would need the impact speed, not the force.
3. **Putdown's two setups are mixed in its metrics.** Each episode starts after pickup (held low) or
   pickup-raised (held high), at random, and the success rate averages both. Record the setup chain with
   each evaluation episode and report them separately.
4. **Run logs are large** (40–80 MB each, mostly progress-bar redraws). Write progress to a small
   structured file, which the runner can read instead.

## 4. Features to add

1. **Analysis without new training**
   - **Transfer gain** in `review_runs.py`: the repaired skill's steps to threshold in a repair run, divided
     by its steps to threshold when learned from scratch. The learnability runs already train every
     held-out skill from scratch.
   - **Confidence intervals** across seeds (bootstrap), instead of mean ± std alone.
2. **Operator relabeling (hindsight relabeling at the symbolic level).** An episode that satisfied another
   operator's effects is a success for that operator: a `push-left` episode that ended in `at-front` is a
   successful `push-front` episode. Store it again under that operator's embedding. This gives every
   operator more successes and directly trains the policy to read the operator. It's easiest with sparse
   rewards; with shaped rewards the relabeled transitions need their rewards recomputed.
3. **Throughput.** Several environments per run (a vectorized environment) and more than one gradient
   update per step (2–4). Both are cheap on a GPU but change the baseline, so apply them to every condition.
4. **Discount for long skills.** `gamma` is 0.95, about a 20-step horizon; the diagonal pushes take ~70
   steps. Try 0.98.
5. **Learned setup chains** (TODO in `skill_environment.py`): set skills up with trained policies instead of
   scripted ones, so training links skills into chains.
6. **Demonstration-seeded replay buffer** (optional; training has been RL only by choice): seed the buffer
   with scripted episodes. It's still RL, but usually much faster.
7. **Better competence sampling.** `--skill-sampling competence` (`competence_shares` in
   `skill_environment.py`) gives episodes to skills in proportion to their recent failure rate. Success is the
   right signal (shaped rewards differ in stages and scale between skills; success means the same for every
   skill and condition), but counting failures misses:
   - **a skill that cannot be learned** keeps a large share forever (push-beside reached only 0.55 alone);
   - **at 0% success** it cannot tell "almost there" from "hopeless";
   - **noise:** 20 episodes per skill, about ±0.1;
   - **resume:** each skill's recent success starts empty, so every skill counts as failing and the shares are
     even for about the first 20 episodes per skill (including a repair's first episodes).

   Options: gate unlearnable skills out of the training set; **learning progress** (favour skills whose
   success is changing: older vs newer half of the window, with a small failure-rate term so a skill at 0%
   still gets episodes), as in Teacher–Student Curriculum Learning (Matiisen et al.) and ALP-GMM (Portelas et
   al.), or the critic's prediction errors, as in Prioritized Level Replay (Jiang et al.); or the recent
   return as a fraction of the scripted policy's (dense, but it inherits the shaped rewards' quirks).
   Learning progress is the most defensible; on resume, each skill's recent success can be read from the
   run's `monitor.csv`.

## 5. Research direction, per question

**RQ1: does conditioning on the operator's structure help a shared multi-skill policy?**
- *Built:* every condition (one-hot, random, text, multi-hot, trainable, compositional tree / slots /
  geometric), the rq1 group, and per-skill steps to threshold, AUC and final success.
- *Next:*
  - Fair, adequately sized runs: 3+ seeds and `RQ1_STEPS=1000000`.
  - A **wrong-embedding control:** evaluate each skill with another skill's embedding. If behaviour doesn't
    change, the policy identifies skills from the state or the grounding, not the operator. Stack, putdown
    and place-beside start from the same state, but their groundings differ in how many slots are used,
    which also identifies the skill.

**RQ2: does structure speed up learning a new skill?**
- *Built:* resuming with added skills, dense-then-sparse evaluation of added skills, and zero-shot points.
- *Next:* the protocol from `research_focus.md`: add skills one at a time in shuffled orders across seeds,
  and measure steps to threshold per new skill. Add it as an experiment (`src/cam/experiments/definitions.py`)
  with its own runner group.

**RQ3: does structure allow repair (zero-shot or faster)?** The most important experiment to run:
`./scripts/start_overnight.sh --experiment rq3-repair`.
- *Built:*
  - the `rq3-repair` experiment (its docstring in `src/cam/experiments/definitions.py` gives the design,
    controls and what to read), with three repair pairs: `unstack` → `unstack-raised`, `stack` → `place-beside`,
    `push-left` → `push-left-front`;
  - every literal of each repaired operator is trained in another skill, so a repair asks how much a policy
    learns about a behaviour from a literal in one place of the operator: `pickup-raised` for `(raised ?o)`;
    `stack`, `putdown` and `push-beside` for place-beside; `push-front` and the other diagonals for
    `(at-front ?o)`. The runner refuses a plan that leaves a literal untrained (e.g. `BASE_SKILLS` without
    `push-beside`);
  - repair-visible and repair-hidden variants, and the invalid-repair flag.
- *Next:*
  - transfer gain (§4.1);
  - more pairs where the repaired literal appears in training in a different skill;
  - `stack` → `place-beside` changes several literals, so read it as a new-skill test;
  - push-beside reached only 0.55 alone in the 2026-10-11 run; if it can't be learned, place-beside's
    `(beside ?o ?u)` is trained but not learned, and that pair says little.

**RQ4: can embeddings generate policies (hypernetworks)?**
- *Built:* nothing yet, apart from the single-skill learnability runs, which are the "individually trained
  policies" baseline.
- *Next:* If I ever ask you to look through here and implement, skip this one for now.
  1. FiLM conditioning: the embedding scales and shifts the state features.
  2. A hypernetwork actor: a custom SB3 SAC actor whose action head weights are generated from the
     embedding, low-rank W(e) = U·diag(h(e))·V.

  This introduces the `PolicyProvider` seam from the redesign doc.

**RQ5: do learned embeddings encode operator structure?**
- *Next:*
  - Linear probes from each learned embedding to its literals, with the multi-hot vector as the target.
  - A 2-D PCA plot of embeddings during training (one existed in the original implementation:
    `plot_embeddings.py`, git commit 59054395).
     - I like this idea, and plotting each literal on the same 2-3 dimensions, so that we can see the relation of e.g. where left, front, back, right, ... lie, as well as things like pickup-putdown vector : stack-unstack vector (which I assume would have similar vector directions, magnitudes)

**RQ6: does behavioural sensitivity reveal unnecessary or incorrect literals?**
- *Next:* If I ever ask you to look through here and implement, skip this one for now.
  - Remove or alter one literal at a time and compare behaviour on fixed probe states.
  - Needs probe-state datasets and action-similarity metrics (action MSE or KL), as in the redesign doc.

**Further out:**
- **Rotation skills.** These need wrist yaw control (a 5-D action variant of the environment) and global
  yaw predicates, e.g. `yaw-0` / `yaw-45` / `yaw-90`. Test generalization on an in-between angle or a chain.
- **A drop skill** as a control: it's structurally identical to putdown but needs different behaviour, so
  structure-based embeddings *should* fail on it.
- **Ideas from the original implementation** (git commit 59054395): behaviour-cloning warm starts, keyboard
  teleoperation with live facts, and a push-rotation penalty.

## 6. Suggested order

1. **Read the current overnight run** with §2.
2. **Before the next overnight run:** add transfer gain and confidence intervals (§4.1), and shorten run
   names (§3.1). Then run `--experiment rq3-repair` (learnability, RQ1, then the repairs) with
   `SEEDS="0 1 2"`.
3. **Then, depending on the results:** operator relabeling (§4.2) and the RQ1 wrong-embedding control, or
   RQ2's protocol, or RQ4 starting with FiLM.
