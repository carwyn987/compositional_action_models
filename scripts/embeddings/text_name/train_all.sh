#!/usr/bin/env bash
set -euo pipefail

# Train + evaluate every skill using a pretrained TEXT embedding of the
# operator NAME alone (e.g. "pickup", "stack").
#
# Defaults to the offline 'hash' backend so it runs with no API key. Switch to
# real pretrained embeddings with:
#   EMBED_BACKEND=openai OPENAI_API_KEY=... ./scripts/embeddings/text_name/train_all.sh
#
# All variables understood by per_skill/train_all_and_eval.sh are forwarded.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env \
  EMBEDDER=text \
  EMBED_SOURCE=name \
  EMBED_BACKEND="${EMBED_BACKEND:-hash}" \
  "${HERE}/../../per_skill/train_all_and_eval.sh" "$@"
