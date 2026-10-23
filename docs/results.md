# Results

The research results: each experiment run on the current environment and rewards, with its date, commit and
settings, what it measured, how it reads against its research question, and what could explain it. The questions
and their hypotheses are in `docs/research_focus.md`; how to run each experiment is in `docs/experiments.md`. The
development runs that shaped the environment, rewards and training (before these) are in `docs/project_status.md`
§5.

A result from one seed is a pilot: it shows a direction and what to test next, not a claim. Steps to threshold are
the training steps until a skill's deterministic success over 20 evaluation episodes reaches 0.95 (each evaluation
is ±0.1 or so), at the resolution of the evaluation interval.

## Validity: can each skill be learned alone?

### Learnability, 2026-10-21

The `learnability` experiment: each skill trained alone by its own multi-task SAC policy (multi-hot embedding),
300k steps, evaluated every 10k steps; shaped rewards, 2 blocks, seed 0, GPU; commit `ffc153f`.

| skill | steps to 0.95 (deterministic) | final success |
| --- | --- | --- |
| unstack | 0: solved untrained (see below) | 1.00 |
| putdown | 20k | 1.00 |
| unstack-raised | 30k | 1.00 |
| place-beside | 70k | 1.00 |
| pickup-raised | 80k | 1.00 |
| pickup | 100k | 1.00 |
| push-back | 150k | 1.00 |
| push-front | 160k | 1.00 |
| push-left | 170k | 0.90 |
| push-right | 170k | 1.00 |
| push-left-front | 240k | 1.00 |
| push-right-back | 250k | 0.80 |
| push-right-front | 290k | 1.00 |
| push-left-back | not within 300k (stochastic: 280k) | 0.90 |
| stack | not within 300k | 0.70 |
| push-beside | not within 300k | 0.60 |

**Reading.** Every skill is learnable alone. The last three were still improving at 300k (stack 0.00 at 200k to
0.70, push-left-back 0.25 to 0.90, push-beside 0.35 to 0.60): they need more steps, and their from-scratch baselines
are lower bounds (more than 300k). The learnability experiment trains for 400k steps (`learn_steps`).

**Unstack was trivial.** Its setup left the open gripper around the top block, and seed 0's untrained network,
which outputs almost the same small action everywhere (close a little, rise), succeeded on every scene. Unstack and
unstack-raised start after a retreat: the open gripper is raised 6–12 cm and moved 5–15 cm sideways, so they
start with an approach, as pickup does. Their learnability with the retreat is not in this table (unstack-raised's
30k is from the easier start); with it, alone, unstack needs 60k steps and unstack-raised 80k (2026-10-22, 200k
steps, seed 0).

## RQ3: does structure allow repairing a skill?

The hypothesis (`docs/research_focus.md`): a content-based operator embedding changes exactly where the operator
changed, so a policy conditioned on it carries out a repaired operator zero-shot more often, or learns it in fewer
steps, than one conditioned on a skill identifier.

### Experiment 1 (pilot), 2026-10-22: unstack-raised added to a policy trained on pickup, unstack and pickup-raised

**Design.** Two conditions, one seed each: one-hot (an identifier) and compositional-tree without the operator's
name (the operator's structure only). Each trains one shared multi-task SAC policy in two phases:

1. pickup, unstack and pickup-raised until each reaches 0.95;
2. the same run with unstack-raised added, training every skill, until unstack-raised reaches 0.95 on two
   evaluations in a row and the other three are back at their level before it was added.

Unstack-raised is unstack with one effect changed, `(not (raised ?o))` to `(raised ?o)`, and every one of its
literals is in a training skill (coverage 1): `(raised ?o)` in pickup-raised, `(on ?o ?u)` and `(clear ?u)` in
unstack. Shaped rewards, 2 blocks, competence sampling, evaluated every 10k steps with 20 episodes (unstack-raised
also 1k, 2k, 4k and 8k steps after it was added), GPU; the code of commit `27ca2b85` (the retreat), run with a
one-off script.
`--experiment rq2-add-unstack-raised` reproduces it. It adds the repaired operator next to its original, as RQ2's
protocol adds skills; RQ3's own (`rq3-repair-minimal`) replaces the original and trains the repaired operator alone.

**Phase 1.** Both conditions learned the three skills in about the same time:

| | one-hot | compositional-tree, no name |
| --- | --- | --- |
| all three at 0.95 | 180k steps | 192k steps |
| steps on each skill until its threshold: pickup, pickup-raised, unstack | 53k, 81k, 42k | 53k, 97k, 25k |

**Phase 2: unstack-raised.**

| | one-hot | compositional-tree, no name |
| --- | --- | --- |
| zero-shot success (deterministic / stochastic) | 0.00 / 0.00 | 0.00 / 0.00 |
| steps to 0.95 after it was added | 90k | 80k |
| of those, steps on unstack-raised itself | 61k | 59k |
| the other three, mean: before, lowest after, final | 0.89, **0.88**, 0.95 | 0.93, **0.52**, 0.98 |
| the other three back at their level after | 20k steps | 20k steps |

Unstack-raised's curve was not monotonic: in the compositional run 0.00 when added and 18k steps later, 0.65 at 58k,
first 0.95 at 80k, 0.65 again at 98k, and 1.00 at 138k, when the run stopped.

**Reading.** The pilot points against the hypothesis:

- **No zero-shot transfer.** Neither condition carried out unstack-raised before training on it, although the
  policy had seen every literal of it.
- **No faster learning.** 80k against 90k steps is one evaluation interval, and the steps on the skill itself
  differ by 4%.
- **More forgetting with structure.** Adding unstack-raised pulled the compositional policy's other skills down to
  a mean lowest of 0.52; one-hot's barely moved (0.88 from 0.89). With a shared composition, the new operator's
  embedding is built from the same components as unstack's and pickup-raised's, so training it moves theirs too;
  one-hot's new skill has an input no other skill uses. Both recovered within 20k steps.

There is no from-scratch baseline with the retreat yet, so the transfer gain is unknown; the 30k steps
unstack-raised needed alone from the easier start are no evidence of a gain.

**What could explain it, other than the hypothesis being false:**

1. **Too few operators to learn what a literal means** (the working hypothesis). A composition learned from three
   operators has had little reason to give a literal a meaning of its own, apart from the operator it came in.
   `(raised ?o)` as an effect is in exactly one registered operator, pickup-raised, always after a grasp from the
   table, so even all 13 training skills give it one context. The literals unstack-raised needs, by the training
   skills that have them:

   | literal of unstack-raised | in |
   | --- | --- |
   | effect `(raised ?o)` | pickup-raised only |
   | precondition `(on ?o ?u)`, effects `(clear ?u)`, `(not (on ?o ?u))` | unstack only |
   | effects `(holding ?o)`, `(not (clear ?o))`, `(not (gripper-empty))` | pickup, unstack, pickup-raised |
   | preconditions `(clear ?o)`, `(gripper-empty)` | pickup, unstack, pickup-raised (and every push) |

   Testing it needs repairs whose new literals several training operators share: the pushes (`(at-left ?o)` and
   `(at-front ?o)` are each in a straight push and two diagonals), or new operators that use `(raised ?o)` in other
   contexts (e.g. a raise and a lower skill on a held block).
2. **How the composition was trained.** With SAC, the actor and critic share the operator embedding's extractor,
   and Stable-Baselines3 trains a shared extractor through the actor's loss only. The composition also had twice
   the configured weight decay (the decay was applied to the shared extractor once for the actor and once for the
   critic; it is applied once from the next runs on).
3. **The behaviour differs where the structure is shared.** `(raised ?o)` asks for lifting to 30–36 cm and holding
   there; pickup-raised learned that lift from a block on the table, while unstack-raised starts from a block on
   another block, a different state the lift has to generalize to.
4. **One seed.** Each evaluation is 20 episodes, and steps to threshold resolve to 10k.

**Next.** Replicate under RQ3's protocol, with seeds, controls and the from-scratch baseline: Experiment 2.

### Experiment 2, 2026-10-23: unstack repaired to unstack-raised, three seeds

**Design.** `rq3-repair-minimal` restricted to this pair: one shared policy per condition and seed on pickup,
pickup-raised and unstack (300k steps), then unstack replaced by unstack-raised, which alone trains (repair mode
isolated) until it reaches 0.95 on two evaluations in a row (at most 150k steps). Conditions: one-hot and
compositional-tree without the operator's name (repair-visible), and the control one-hot repair-hidden, where
unstack-raised gets exactly unstack's code. Seeds 0, 1 and 2; the from-scratch baseline is unstack-raised's
learnability run with the retreat (80k steps). Settings as Experiment 1, commit `6731e4f5` (composition weight
decay applied once); intervals are the bootstrap 95% intervals of the mean over the three seeds, which with three
seeds are the seeds' range.

**The repair.**

| condition | zero-shot success (deterministic) | steps to 0.95 | transfer gain (80k / steps) |
| --- | --- | --- | --- |
| one-hot, repair-visible | 0.02 [0, 0.05] | **37k [30k, 40k]** | **2.2×** |
| compositional-tree, no name, repair-visible | 0.00 | 77k [50k, 100k] | 1.0× |
| one-hot, repair-hidden (control) | 0.00 | 120k [110k, 140k] | 0.7× |

The other skills (evaluated, not trained) kept their level in every condition: mean lowest after 0.94 [0.88, 0.98]
compositional, 0.98 for both one-hot variants, back within 20k steps.

**The shared policies before the repair** (RQ1 on three skills, 300k steps):

| | one-hot | compositional-tree, no name |
| --- | --- | --- |
| steps to 0.95: pickup, pickup-raised, unstack | 200k, 200k, 150k | 183k, 183k, ≥217k (seed 1 never) |
| final success, unstack | 1.00 in every seed | 1.00, 0.00 (seed 1: 0.00 at every evaluation), 0.95 |

**Reading.**

- **The pair is valid.** Repair-hidden fails zero-shot (0.00): unstack's behaviour does not satisfy unstack-raised.
  With unstack's code it then takes longer than from scratch (120k against 80k), so its steps are no speed
  baseline. Its input is exactly unstack's (code, scenes and grounding), so the replay buffer it starts from (the
  rq1 run's, about a third unstack) holds transitions with the same inputs and the opposite reward for lifting into
  the raised band, until new ones replace them; it also keeps unstack's converged entropy temperature (about 0.05;
  multi-task SAC keeps one per embedding, and a new embedding's starts at 1), and the critic's and actor's
  confident unstack values have to be unlearned. A repaired skill that keeps its identity is repaired more slowly
  than one learned anew, which an embedding that changes with the operator avoids.
- **The identifier repairs fastest.** With a code no training used, one-hot learns unstack-raised in 37k steps,
  2.2 times faster than from scratch: what the shared policy learned on the three skills transfers, through the
  state and the network, not through the operator.
- **The compositional embedding gives no gain** (77k, the from-scratch 80k), and is slower than one-hot: the seeds'
  ranges do not overlap ([50k, 100k] against [30k, 40k]). No condition succeeds zero-shot.
- **It learned the three skills no faster,** and in one seed of three not unstack at all.

**Why: the compositional embedding collapsed.** Review section 6 shows, in all three repaired compositional
policies, unstack-raised's embedding identical to pickup-raised's (cosine 1.000, distance under 1e-4): the policy
was given pickup-raised's input for unstack-raised and could tell them apart only from the state. Before the repair,
pickup and unstack were already nearly identical (cosine 0.91, 0.98 and 0.99 in seeds 0–2; seed 1 never learned
unstack). The cause is the composition's scale: its output before the extractor normalizes it grew from about 0.8
to 1e3–1e10 during training (`operator_embedding_magnitude_before_normalization`, flagged "exploding" in section 4),
largest for the operators with `(raised ?o)` (pickup-raised 1e6–7e9). One component then dominates every operator
it is in, and the normalization, which divides by the total, hides the growth from the loss: operators that share
the dominant literal become one point, whatever else differs. The composition weight decay (0.01) does not hold the
scale. So the compositional results so far measure a collapsed embedding, not the operator's structure; the same
growth is likely in Experiment 1 (not checked there).

**Next.** Bound the composition's scale (`docs/next_steps.md` §3.4), check on short trainings that the embeddings of
pickup, pickup-raised, unstack and unstack-raised stay apart, and run this experiment again: the comparison with
one-hot's 37k is then the test of RQ3 on this pair. Then the working hypothesis of Experiment 1.
