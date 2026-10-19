# Compositional action models

Robot skills learned with reinforcement learning, where one shared policy is conditioned on an embedding
of each skill's symbolic action model (a PDDL operator). The research questions: does the operator's
structure make multi-skill learning faster, help learn new skills, and allow repairing a skill when its
operator changes, compared with arbitrary skill identifiers? The skills are blocksworld actions (pickup,
putdown, stack, unstack, place-beside, pushes) on a Gymnasium-Robotics Fetch arm with 1–5 blocks, trained
with Stable-Baselines3 SAC or PPO.

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
./scripts/review_runs.py outputs/overnight_<date>            # report: runs, conditions, curves, repairs
./scripts/watch_best_policies.py                             # the best trained policy per skill
LOGDIR=outputs/overnight_<date> ./scripts/start_tensorboard.sh
```

Each script's header and `--help` describe its options. `main.py --help` lists the operator embedding
conditions (`--operator-encoder`, `--trainable-operator-embedding`, `--compositional-architecture`).

## Documentation

| file | contents |
| --- | --- |
| `docs/research_focus.md` | the research questions (RQ1–RQ6), their experiments and metrics |
| `docs/related_work.md` | the closest prior work for each research question, what this project adds, and references |
| `docs/project_status.md` | what the code does: predicates, skills, success and failure, embeddings, metrics, scripts, and results so far |
| `docs/next_steps.md` | the backlog: open issues, features, the direction for each research question; start a new session here |
| `docs/policy_inputs.md` | the policy's observation: state, operator embedding (multi-hot), grounding slots |
| `docs/operator_embeddings.md` | every operator embedding condition, with worked examples |
| `docs/component_diagrams.md` | component map, one training episode, class diagram (`.html` renders them in a browser) |
| `docs/repository_redesign_overview.md` | the target architecture, and what of it is built |

## Use of AI

The code, documentation and analyses in this repository were created with AI assistance (Anthropic's Claude,
through Claude Code), under strict human instruction, guidance and evaluation. The author sets the research
questions and experiment designs, directs each change, and reviews, tests and evaluates every result commit by
commit before it is merged.
