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
