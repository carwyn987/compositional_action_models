# scripts/

Training/evaluation entrypoints for the embedding-augmented skills.

## `train_all_methods.sh`

Trains a **single shared model on several skills at once** (default
`pickup putdown`, set via `SKILLS`), where the only signal distinguishing the
skills to the policy is the symbolic embedding. It does this once per embedding
scheme — `mock`, `name`, `full` — **and** per mode (`frozen`, `trainable`), i.e.
six runs by default, all sharing identical hyperparameters (set via env vars;
see the file header). Vanilla is excluded: with no embedding the shared policy
cannot tell the skills apart. Drives `train_multiskill.py`.

### Embedding modes

- **frozen** — the per-skill embedding table is initialised from the pretrained
  embeddings and held fixed.
- **trainable** — the same table is a policy parameter, learned during BC + RL.
  The learned values are **never written back** to the pretrained embedding
  source (`symb_model_embeddings/embedding_cache.json`); they live only in the
  saved policy checkpoint.

Every run records its embedding table at the start and at intervals during
training to `embedding_logs/<run>.json` (configurable via `EMBED_LOG_DIR`).
Visualise how the embeddings begin and move with:

```
python plot_embeddings.py --log-dir embedding_logs
```

which projects all embeddings to 2-D (PCA): frozen runs show one point per
skill, trainable runs show a trajectory (hollow `o` = initial, `★` = learned),
coloured by scheme and labelled by mode.

## `tensorboard.sh`

Launch TensorBoard on the training logs (`LOGDIR`, default `runs/`).

## `per_skill/`

Hand-tuned, per-skill recipes plus the shared sweep driver:

- `train_all_and_eval.sh` — trains + evaluates every skill. Configurable via
  environment variables (see the header in the file), including the `EMBED_*`
  variables that select the symbolic embedding. This is the driver the
  per-embedding wrappers below call.
- `train_pickup.sh`, `train_push.sh`, `train_putdown.sh`, `train_stack.sh`,
  `train_unstack.sh` — individual tuned recipes.
- `vis_push.sh` — render the trained push policies.

## `embeddings/`

One folder per embedding type. Each `train_all.sh` is a thin wrapper that sets
the `EMBED_*` variables and forwards everything else to
`per_skill/train_all_and_eval.sh`:

- `mock/` — fixed random per-skill vector (ignores symbolic content; baseline).
- `text_name/` — pretrained text embedding of the operator *name*.
- `text_action_model/` — pretrained text embedding of the full PDDL-style
  *action-model* string.

The text variants default to the offline `hash` backend (no API key needed).
Set `EMBED_BACKEND=openai` (with `OPENAI_API_KEY`) for real pretrained
embeddings.

## Vanilla control

Scripts that train the skills **without** any symbolic embedding live under
`../experimental/scripts/` and drive the vanilla entrypoints.

## Model / log file names

Names encode the major option choices that change the trained artifact, so runs
with different settings never overwrite each other:

```
{skill}_{algo}_{embed_tag}_t-{teacher}_s{seed}
```

where `embed_tag` is `mock-d{size}`, `text-{source}-{backend}-d{size}`, or
`vanilla` (no embedding). Examples:

- `pickup_ppo_mock-d32_t-bc_s0.zip`
- `pickup_ppo_text-action_model-hash-d32_t-bc_s0.zip`
- `pickup_ppo_vanilla_t-bc_s0.zip`

The naming is defined once in `symb_model_embeddings/naming.py`; the bash drivers
call `python -m symb_model_embeddings.run_naming ...` so model, eval, and log
names always agree.
