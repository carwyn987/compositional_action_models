# scripts/

Training/evaluation entrypoints for the embedding-augmented skills.

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
