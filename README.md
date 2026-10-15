# Compositional action models

Robot skills learned with reinforcement learning, where one shared policy is conditioned on an embedding
of each skill's symbolic action model (a PDDL operator: parameters, preconditions, effects). A planner
already reasons with these operators; the question is whether their structure can also tell a learned
low-level policy what to do. Compared with arbitrary skill identifiers, does the operator's structure make
multi-skill learning faster (RQ1), help learn new skills (RQ2), and allow repairing a skill when its operator
changes (RQ3)? Can a hypernetwork generate a skill's policy from its operator embedding (RQ4)? And do learned
embeddings encode the operator's structure (RQ5), or reveal unnecessary or incorrect literals (RQ6)?

The skills are blocksworld actions (pickup, putdown, stack, unstack, place-beside, pushes) on a
Gymnasium-Robotics Fetch arm with 1–5 blocks, trained with Stable-Baselines3 SAC, multi-task SAC or PPO.

## Install

Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU; CPU: .../whl/cpu
pip install -r requirements.txt
pip install -e .
```

The text embedding condition with `--text-backend openai` reads the OpenAI key from `OPENAI_API_KEY` or
`~/.secrets/openai_api_key`. The default `mock` backend needs no key.

## Quickstart

```bash
./run_tests.py -m unit                                       # fast tests; ./run_tests.py runs all

./scripts/run_skill.py --skill stack --render                # watch a skill's scripted policy
./scripts/run_skill.py --skill push-left --episodes 20       # its success rate, no window

./main.py --skills pickup putdown --total-timesteps 200000 --reward shaped    # train and evaluate
./main.py --run-directory outputs/<run>                      # resume a run
./scripts/run_skill.py --policy model --run-directory outputs/<run> --skill pickup --render

DRY_RUN=1 ./scripts/start_overnight.sh                       # the overnight plan and its time estimate
DEVICE=cuda ./scripts/start_overnight.sh                     # run it (inside tmux)
RUN_GROUPS=learnability JOBS=6 ./scripts/start_overnight.sh  # one group, 6 runs at a time
CONDITIONS="one-hot compositional-tree" SEEDS="0 1 2" ./scripts/start_overnight.sh   # settings: see overnight_runs.py
./scripts/review_runs.py outputs/overnight_<date>            # report: runs, conditions, curves, repairs
./scripts/watch_best_policies.py                             # the best trained policy per skill
LOGDIR=outputs/overnight_<date> ./scripts/start_tensorboard.sh
```

Each script's header and `--help` describe its options; the overnight runner's settings are environment
variables listed in the header of `scripts/overnight_runs.py`. `main.py --help` lists the operator embedding
conditions (`--operator-encoder`, `--trainable-operator-embedding`, `--compositional-architecture`).

## Documentation

Read the first three in order: the research, then what is built and found, then what is next. The others
are reference.

| file | contents |
| --- | --- |
| `docs/research_focus.md` | the research questions (RQ1–RQ6): for each, the hypothesis, experiment, controls, metrics and interpretation |
| `docs/project_status.md` | what the code does (environment, predicates, skills, success and failure, embeddings, training, metrics, scripts) and the results so far |
| `docs/next_steps.md` | the single backlog: open issues, features, the next work for each research question; start a new session here |
| `docs/policy_inputs.md` | the policy's observation: state, operator embedding, grounding slots |
| `docs/operator_embeddings.md` | every operator embedding condition (one-hot, random, text, multi-hot, trainable, compositional), with worked examples |
| `docs/component_diagrams.md` | component map, one training episode, class diagram (`.html` renders them in a browser) |
