# Experiments

How to run each experiment, what it answers, and where its results are. The research questions themselves
are in `docs/research_focus.md`.

An experiment is a class in `src/cam/experiments/research_experiments.py`. Its docstring says what it asks,
how, and which metrics answer it; its attributes are the settings of the overnight runner
(`scripts/overnight_runs.py`), which plans and runs its training runs. This file summarizes them; the
docstrings and the runner's header are the reference. `./scripts/start_overnight.sh --list` prints every
experiment's docstring and settings.

## Running an experiment

```bash
DRY_RUN=1 ./scripts/start_overnight.sh --experiment NAME   # the plan and its time estimate; nothing runs
./scripts/start_overnight.sh --experiment NAME             # run it in this terminal (inside tmux)
```

`start_overnight.sh` runs the unit tests, records the commit and the settings in `<OUT>/run_info.txt`, then
runs the plan. Closing its terminal stops it. To run it in the background, immune to the editor or the
terminal closing:

```bash
systemd-run --user --unit=NAME --same-dir --setenv=OUT=outputs/overnight_<date> --setenv=DEVICE=cuda \
    ./scripts/start_overnight.sh --experiment NAME
journalctl --user -u NAME -f            # follow its output
systemctl --user kill -s INT NAME       # stop it: every running run saves first
```

A systemd unit does not see the shell's environment: pass each setting with `--setenv` (the OpenAI key is
read from `~/.secrets/openai_api_key`). A unit that failed keeps its name until
`systemctl --user reset-failed NAME`.

Starting the same command again with the same `OUT` continues the plan: finished runs are skipped and
interrupted runs resume from their last save. Before anything trains, the runner checks the device, embeds and
caches every operator text the runs need (`TEXT_BACKEND=openai` needs the key and the network) and runs every
skill's scripted policy.

### Settings

Every attribute of an experiment is a setting, overridden for one invocation by an environment variable of
its name in upper case; the runner lists the overrides it used. The ones most often changed:

| setting | what it sets |
| --- | --- |
| `SEEDS` | the seeds of the rq1 and repair runs, e.g. `SEEDS="0 1 2"`: a claim needs several |
| `OPERATOR_EMBEDDINGS` | the conditions compared, e.g. `"one-hot compositional-tree"` |
| `OPERATOR_NAMES` | whether text and compositional embeddings see the operator's name: `exclude` (default), `include`, or `"exclude include"` (see Operator names) |
| `BASE_SKILLS`, `REPAIRS` | the training skills, and the repairs as `FROM:TO` pairs |
| `REPAIR_MODES` | `isolated` (only the repaired skill trains), `retain` (every skill trains), or both |
| `LEARN_STEPS`, `RQ1_STEPS`, `REPAIR_STEPS` | the budgets; `RQ1_STEPS=auto` sets the rq1 budget from the learnability runs |
| `RQ2_ORDERS`, `RQ2_SHUFFLES` | rq2's orders, stages separated by `,` and a stage's skills joined by `+` (e.g. `"pickup+unstack+pickup-raised,unstack-raised"`); or that many shuffles of `BASE_SKILLS`, one skill per stage |
| `RQ2_FIRST_STEPS`, `RQ2_STAGE_STEPS` | an rq2 chain's budgets: its first stage, and each later one |
| `TEXT_BACKEND` | `openai`, or `mock` (offline, no meaning) |

Machine settings: `OUT` (default `outputs/overnight_<date>`), `JOBS` (runs at once, default 2), `DEVICE`
(`auto`, `cpu`, `cuda`), `THREADS`, `DRY_RUN`.

### Outputs and the review

Under `OUT`:

| path | contents |
| --- | --- |
| `learnability/<skill>/<run name>/` | a learnability run: `config.json`, `metrics.json`, `model.zip`, `tensorboard/`, `train.log` |
| `rq1/<condition>_seed<n>/<run name>/` | an rq1 run |
| `repair/<TO>/<condition>_<repair-visible or repair-hidden>_<mode>_seed<n>/` | a repair: a copy of the rq1 run, continued |
| `rq2/<condition>_seed<n>/<order>/<run name>/` | an rq2 chain: one run, continued stage by stage |
| `logs/<label>.log`, `failed.txt` | each run's full output; the runs that failed |
| `overlap.txt`, `run_info.txt` | the overlap report of the experiment's skills; the commit and settings |
| `summary.csv`, `review.txt` | per run and skill metrics; `scripts/review_runs.py`'s report, written when the plan ends |

`./scripts/review_runs.py <OUT>` writes the report at any time, also while runs go: 1 runs, 2 conditions
(per skill, mean ± std over seeds), 3 curves, 4 training statistics, 5 added and repaired skills, 6 learned
embeddings. `LOGDIR=<OUT> ./scripts/start_tensorboard.sh` shows the curves, and
`./scripts/watch_best_policies.py --folder <OUT>` plays the best policy for each skill.

## The experiments

### `learnability`: can each skill be learned alone, and how fast?

Every registered skill trained alone by its own policy. A skill that cannot be learned alone makes any
result that uses it meaningless, so this is the gate for the other experiments; its steps to threshold are
also the from-scratch baseline of a repair's transfer gain and what `RQ1_STEPS=auto` adds up. Read
`review.txt` section 2 (steps to threshold, final success) and section 3 (curves); for a skill that stays
low, the docstring says where to look next. Settings: `LEARN_STEPS`, `LEARN_EVAL_INTERVAL`,
`LEARNABILITY_EMBEDDING`.

### `rq1`: does conditioning on the operator's structure help a shared policy?

The learnability runs, then one shared policy on the 13 training skills per operator embedding and seed:
identifiers (one-hot, random) against the operator's literals (multi-hot, text, compositional). Read section
2 per skill, not averaged, across seeds (`SEEDS="0 1 2"`): steps to threshold in the skill's own steps, AUC
and final success; next to each skill's shared fraction in `overlap.txt`. Settings: `OPERATOR_EMBEDDINGS`,
`SEEDS`, `RQ1_STEPS`, `OPERATOR_NAMES`.

### `rq3-repair-minimal` and `rq3-repair-maximal`: does structure let a policy carry out a repaired operator?

The learnability and rq1 runs, then each rq1 policy continued with a training skill (FROM) replaced by a
held-out operator (TO) every literal of which another training skill has; TO is evaluated zero-shot, then
trained alone until it reaches the threshold. Controls: repair-hidden (TO gets FROM's exact embedding, for
one-hot and random), one-hot's unused index, and operator embeddings without the operator's name.

- `rq3-repair-minimal`: seven training skills and three repairs (unstack → unstack-raised, putdown →
  place-beside, push-left → push-left-front), each repair's new literals in exactly one other skill. Read it
  for a faster repair more than for zero-shot success.
- `rq3-repair-maximal`: the 13 training skills and push-left → push-left-front, whose new literals are trained
  in several pushes. The most favourable case for zero-shot success, and the longest rq1 runs.

Read section 5: TO's zero-shot success, repair-visible against repair-hidden (`REPAIR INVALID` flags a pair
whose hidden variant succeeds), its steps to threshold against its learnability run (the transfer gain), and
what the other skills forgot. Check first, in section 2, that the rq1 policy learned FROM and the skills that
carry TO's literals. Settings: `REPAIRS`, `REPAIR_MODES` (`"isolated retain"` adds the cost of a repair
without forgetting), `REPAIR_STEPS`, `OPERATOR_EMBEDDINGS`, `SEEDS`, `OPERATOR_NAMES`.

### `rq2` and `rq2-add-unstack-raised`: does structure speed up learning the next skill?

One shared policy per operator embedding, order and seed, given skills a stage at a time. Stage 1 is a new run
on its skills until each reaches the threshold; each later stage continues that run with its skills added
(repair mode retain), until they reach the threshold and the earlier skills are back at their level. A stage
starts only once the one before reached the threshold, otherwise the chain ends there (section 1 of the review
says why). For each added skill: does it succeed zero-shot (possible only where the stages before cover all
its literals: the plan prints each added skill's coverage), does it reach the threshold in fewer steps than
with one-hot and than from scratch, and what do the earlier skills forget?

- `rq2`: pickup, unstack, pickup-raised and unstack-raised, one at a time, in three shuffled orders (the same
  for every embedding and seed), after a learnability run of each skill (the transfer gain's baseline). Each
  skill's literals are all in the two it differs from in one way (on the table or a block, held low or raised),
  so the last skill of every order has coverage 1.
- `rq2-add-unstack-raised`: one-hot and compositional-tree, side by side: pickup, unstack and pickup-raised
  together until each reaches 0.95 (at most 1.5M steps), then unstack-raised added (at most 1M steps), whose
  literals the three cover. Saved every 20k steps with the replay buffer. No learnability runs, so no transfer
  gain.

Read section 5 (one entry per addition, in the order added, each with its own retention) and section 1.
Settings: `RQ2_ORDERS`, `RQ2_SHUFFLES`, `RQ2_FIRST_STEPS`, `RQ2_STAGE_STEPS`, `OPERATOR_EMBEDDINGS`, `SEEDS`,
`OPERATOR_NAMES`.

```bash
DRY_RUN=1 ./scripts/start_overnight.sh --experiment rq2       # the plan, with each added skill's coverage
systemd-run --user --unit=rq2 --same-dir --setenv=OUT=outputs/rq2_$(date +%Y%m%d) --setenv=DEVICE=cuda \
    ./scripts/start_overnight.sh --experiment rq2
```

### `overlap`: how much of their operators do the training skills share?

Static, nothing trains: the fraction of each skill's predicates, atoms, literals, preconditions and effects,
and whole operator that another training skill shares, and how much of each held-out operator the training
skills cover. Every experiment prints the same report for its own skills when it starts (`overlap.txt`).

```bash
.venv/bin/python scripts/overnight_runs.py --experiment overlap
BASE_SKILLS="pickup unstack pickup-raised" REPAIRS="unstack:unstack-raised" \
    .venv/bin/python scripts/overnight_runs.py --experiment overlap
```

## Operator names

Two operator embeddings carry the operator's name:

- **text** embeds the operator's text, whose PDDL starts `(:action <name>` and whose paragraph starts
  `Action <name> with parameters`;
- **compositional** (`compositional-tree`, `-slots`, `-geometric`) has a name component, the name's text
  embedding, among the components it composes.

One-hot and random are identifiers: the name is their identity by design. Multi-hot has the literals only.

**The experiments leave the name out** (`OPERATOR_NAMES=exclude`, the default). Held-out operators share
words of their names with training skills: unstack-raised with pickup-raised ("raised"), push-left-front with
push-front ("front"), place-beside with push-beside ("beside"). A text model places such names close together
(the mock backend too: they share character n-grams), so with the name a zero-shot success could come from a
word rather than from the operator's structure, which is what RQ3 tests. Without the name, the text calls
every operator `unnamed` and the compositional embedding has no name component: two operators get different
embeddings only where their preconditions, effects or parameters differ.

The setting is `operator_names` in `research_experiments.py`. Each value is a variant of every text and
compositional condition of the rq1, repair and rq2 runs:

| `OPERATOR_NAMES` | runs | main.py options |
| --- | --- | --- |
| `exclude` | `<condition>-no-name`, e.g. `rq1/text-no-name_seed0/` | `--operator-text-name none`, `--compositional-name none` |
| `include` | `<condition>`, e.g. `rq1/text_seed0/` | none: main.py's default includes the name |
| `"exclude include"` | both, side by side | |

A label without `-no-name` always means the name was included, so the variants never mix in a folder,
`review.txt` or `summary.csv`. Other conditions run once whatever the setting. Learnability runs use
`LEARNABILITY_EMBEDDING` as main.py builds it: one skill alone, the name tells the policy nothing.

**The ablation:** how much does the name add? Run the experiment with both variants:

```bash
OPERATOR_NAMES="exclude include" ./scripts/start_overnight.sh --experiment rq3-repair-minimal
```

With the `OUT` of a finished experiment, this trains only the runs with the name: the others are done. Then
compare `<condition>` with `<condition>-no-name` in `review.txt`: section 2 for training, section 5 for the
repairs. A higher zero-shot success with the name is what the name gave.
