# Research focus

The research questions, and for each: the hypothesis, the experiment, the metrics and how results are read.
What the code does and what the experiments have shown so far is in `docs/project_status.md`; how to run each
experiment is in `docs/experiments.md`; what to do next is in `docs/next_steps.md`.

## The question

A symbolic action model (here a PDDL operator: parameters, preconditions, effects) says what a skill
achieves, and a planner can chain operators into plans. A robot still needs a low-level policy to carry out
each one. This project asks: **can a neural network map an embedding of a symbolic action model to a
low-level policy, and does that mapping generalize to embeddings of operators it was not trained on, such as
new skills or repaired operators?**

An operator embedding can reach a policy in three ways:

1. **Conditioning:** one shared policy takes the embedding as part of its input (RQ1–RQ3).
2. **Modulation:** the embedding scales and shifts the shared policy's features (FiLM).
3. **Generation:** a hypernetwork turns the embedding into a policy's parameters (RQ4).

RQ1–RQ4 measure what the embedding does for learning. RQ5 and RQ6 are discussion questions about what the
learned embeddings contain.

## Shared design

### Independent variable: the operator embedding condition

Ordered from no information about the operator to its full structure. The condition names are those of the
overnight runner (`scripts/overnight_runs.py`); `docs/operator_embeddings.md` describes how each is computed.

| kind | condition | what the policy is told about the skill |
| --- | --- | --- |
| identity only | `one-hot` | which skill it is, as a skill ID |
| | `random` | a fixed random vector per skill: an arbitrary identifier |
| | `random-trainable` | a learnable vector per skill, starting random |
| structure, fixed | `text` | the whole operator as text (PDDL or a paragraph), embedded by a text model |
| | `multi-hot` | the operator's literals, by parameter position |
| structure, learned | `multi-hot-trainable` | a learned vector per literal, summed over the operator's literals |
| | `compositional-tree`, `-slots`, `-geometric` | learned components (predicates, variables, types) composed by a learned function that follows the operator's structure |

The text and compositional embeddings can also carry the operator's name, whose words alone can relate two
skills (unstack-raised and pickup-raised share "raised"). The experiments leave it out and run these
conditions as `<condition>-no-name`; the variants with the name are an ablation (`docs/experiments.md`,
Operator names).

### Held fixed between conditions

The policy network (only its operator input differs), the skill set, the order in which training episodes
choose skills, the RL algorithm and its settings, the training budget, the embedding size (256 for every
condition) and the seeds.

### Metrics

Defined in `docs/project_status.md` (Metrics): success rate, **steps to threshold** (training steps until a
skill's success rate reaches 0.95, counted in the skill's own steps), **zero-shot success** (before any
training on the skill), **final success**, **AUC** (mean success rate over the run) and **retention** (what
other skills lose when one is added). **Transfer gain** compares steps to threshold with learning the same
skill from scratch.

Every metric is reported per skill as well as aggregated: an aggregate can be dominated by easy skills (one
skill, putdown, succeeds without training). Comparisons need several seeds.

### Validity checks

- Each skill's scripted policy succeeds, which checks its setup, predicates and success condition.
- Each skill can be learned alone (the learnability runs). A skill that cannot be learned alone says nothing
  about the embeddings.

## RQ1. Does structure help a shared policy learn several skills?

**Question.** Does conditioning a shared multi-skill policy on a representation of each skill's symbolic
action model improve sample efficiency or final performance, compared with arbitrary skill identifiers?

**Hypothesis.** Operators share literals: pickup and unstack both add `(holding ?o)`; push-left and
push-left-front both add `(at-left ?o)`. An embedding built from the operator shows the policy what skills
have in common, so experience on one skill helps the skills that share its literals. Structured conditions
then reach the success threshold in fewer steps, and possibly at a higher final success rate, than
identity-only conditions.

**Experiment.** For each condition, train one shared policy on the same skill set (the overnight runner's
`rq1` group), with everything in "Held fixed" equal, over several seeds.

**Controls.**
- **Identity-only conditions** (`one-hot`, `random`): the same joint training on every skill, without the
  operator's structure.
- **Learnable identity vs learned structure** (`random-trainable` vs `multi-hot-trainable` and
  `compositional-*`): both learn a vector per skill. Only the structured ones build it from the operator's
  parts, which separates "structure as a prior" from "a learned embedding".
- **Wrong embedding:** evaluate each skill with another skill's embedding. If behaviour does not change, the
  policy identifies the skill from the state or the grounding instead of the operator. (Stack, putdown and
  place-beside start from the same state, but their groundings use different numbers of slots, which also
  identifies the skill.)

**Metrics.** Steps to threshold, AUC and final success, per skill and aggregated.

**Interpretation.**
- The shared network learns from every skill in every condition; only the operator input differs. If
  structured conditions reach the threshold in fewer steps, the operator's structure speeds up learning.
  This need not mean the policy generalizes: it shows that, given the structure, skills trained together
  help each other more.
- If structured conditions reach a higher final success rate, the structure leads the network to shared
  pathways between skills that improve performance, not only speed.
- If learned structured embeddings beat the learnable identity vector, the symbolic structure is a useful
  prior, not just extra parameters.
- When every condition can learn every skill, final success saturates and the comparison rests on steps to
  threshold and AUC.

## RQ2. Does structure speed up learning a new skill?

**Question.** How much does conditioning a shared policy on operator embeddings speed up learning the next
skill, and can the new skill succeed zero-shot?

**Hypothesis.** A new operator built from literals the policy has already seen gets an embedding related to
the skills that share them, so the policy starts closer to the new behaviour. With an identity condition the
new skill's code is unrelated to anything trained.

**Experiment.** Repeat over shuffled skill orders and seeds:
1. Shuffle the skills.
2. Train on them in order, adding one skill at a time and keeping every earlier skill in training, so that
   none is forgotten: [1], then [1, 2], then [1, 2, 3], and so on.
3. For each new skill, measure its zero-shot success when it is added and its steps to threshold; reaching
   the threshold ends that stage.

Compare whole-operator text embeddings, aggregate compositional embeddings (one vector composed from the
operator's literals) and slot-based compositional embeddings, with the identity conditions as baselines. Aggregate over orders and seeds for
statistical significance.

The overnight runner's `rq2` experiment (`src/cam/experiments/research_experiments.py`, which gives its design and
the metrics to read) runs this on four skills, the corners of a square: pickup, unstack, pickup-raised and
unstack-raised (on the table or on a block, held low or raised), in shuffled orders, one skill per stage. Each
skill's literals are all in its two neighbours, so in every order the last skill added has coverage 1 by the skills
before it, which zero-shot success needs; an earlier one may lack a literal. `rq2-add-unstack-raised` adds a
single skill: unstack-raised, to a policy trained on the other three together. A stage starts only once the one
before reached the threshold, and continues the same run (`--repair-mode retain`, so that no earlier skill is
forgotten).

**Metrics.** The new skill's zero-shot success and steps to threshold; retention of the earlier skills.

**Interpretation.** How far the shared policy generalizes to new skills, including zero-shot. Retention shows
what adding a skill costs the others.

## RQ3. Does structure allow repairing a skill?

**Question.** When a skill's operator is repaired (an effect or precondition added, removed or changed), does
conditioning on operator embeddings let the policy carry out the repaired operator zero-shot, or learn it
faster?

**Hypothesis.** A content-based embedding changes exactly where the operator changed, so the policy can reuse
what it knows about the unchanged literals and about the changed literal from other skills. Structured
conditions then succeed zero-shot more often, or reach the threshold in fewer steps, than identity
conditions.

**Experiment.** Train a shared policy on a set of skills (an RQ1 run), replace one operator with its repaired
version, evaluate the repaired skill zero-shot, then train until it reaches the threshold. Compare with the
one-hot and random conditions. Each repaired operator uses only literals that appear in some training skill,
so a policy that reads the operator can know them. The run plan sets the repair pairs; the overnight
runner's defaults are:
- `unstack` → `unstack-raised`: one effect changes, `(not (raised ?o))` to `(raised ?o)`;
- `push-left` → `push-left-front`: an effect and a precondition are added, for `at-front`;
- `stack` → `place-beside`: several literals change, so this is closer to a new skill (RQ2).

**Variants and controls.**
- **Repair-visible vs repair-hidden.** Repair-hidden gives the repaired operator exactly the original's
  embedding (`--operator-identity-aliases`, one-hot and random only), so the policy cannot see the repair. If
  repair-hidden succeeds zero-shot, the original behaviour already satisfies the repaired operator, and the
  pair does not test repair.
- **Isolated vs retain.** Isolated trains only the repaired skill, so its steps compare directly with learning
  it from scratch. Retain trains every skill until the repaired one is at the threshold and every other skill
  is back at its earlier level: the cost of a repair without forgetting.
- **From scratch.** The repaired skill learned alone (its learnability run) is the baseline for transfer gain.
- **Without the operator's name.** Held-out operators share words of their names with training skills
  (unstack-raised with pickup-raised, push-left-front with push-front, place-beside with push-beside), so the
  text and compositional embeddings leave the name out: a zero-shot success then comes from the operator's
  structure. Running them with the name too measures what the name adds.

**Metrics.** The repaired skill's zero-shot success, its steps to threshold and transfer gain; retention of
the other skills.

**Interpretation.** Zero-shot success with repair-visible but not with repair-hidden means the policy reads
the changed literals. Fewer steps to threshold than identity conditions and than learning from scratch means
the structure transfers. RQ2 and RQ3 together measure the effect of structure on learning: new skills and
repaired ones.

## RQ4. Can operator embeddings generate policies?

**Question.** Does conditioning individual policy networks on symbolic action models through a hypernetwork
outperform, or give useful initializations compared with, training individual policy networks from scratch?

**Experiment.** Build the pipeline operator → embedding → hypernetwork → policy parameters, and compare the
generated policies with individually trained ones (the learnability runs: each skill learned alone), both as
they are and as the starting point for further training.

**Metrics.** Steps to threshold, from the generated initialization and from scratch; the generated policies'
success before further training.

**Interpretation.** If generated policies succeed, or reach the threshold in fewer steps than training from
scratch, the operator embedding carries enough to specify a policy, and a new operator yields a policy
without training one from scratch.

## Discussion questions

### RQ5. Do learned embeddings encode the operator's structure?

**Experiment.** Try to recover the PDDL operator from its learned embedding. A decoder (for example an
autoregressive autoencoder) may need more operators than the project has; linear probes from each embedding
to its literals, with the multi-hot vector as the target, need fewer.

**Interpretation.** If the operator can be recovered, the learned embedding is a vectorization of the action
model. A hand-built encoding (multi-hot) already vectorizes operators within fixed limits on arity and
predicates, so the question is whether embeddings learned from behaviour keep that structure or reduce to a
skill identity.

### RQ6. Does behavioural sensitivity reveal unnecessary or incorrect literals?

**Experiment.** Remove or change one precondition or effect at a time, condition (or generate) the policy on
the modified operator, and compare its behaviour on fixed probe states (action MSE or KL); then test the
simplified operators these comparisons suggest.

**Open questions.** This one is exploratory. Do operators that are similar in structure but need very
different behaviour end up close together in embedding space? And when does the policy's insensitivity to a
literal show that the literal is unnecessary (for example a precondition that never changes behaviour), as
opposed to the policy ignoring it?

## Not pursued (future work)

- **One conditioned network vs individual networks:** does a single network conditioned on symbolic action
  models, performing every skill, outperform an individual network per skill?
- **Concurrent vs sequential multi-skill training:** does training skills together improve stability or
  transfer compared with training them one after another?
- **Other kinds of action model:** declarative (PDDL) versus procedural or operational models (refinement
  action models), and other model types, such as finite-state machines or individually trained networks used
  as a skill's specification.
