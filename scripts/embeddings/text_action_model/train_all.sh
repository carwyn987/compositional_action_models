#!/usr/bin/env bash
set -euo pipefail

# Train + evaluate every skill using a pretrained TEXT embedding of the full
# PDDL-style ACTION MODEL string (parameters, preconditions, effects) carried
# by each SkillSpec.
#
# Defaults to the offline 'hash' backend so it runs with no API key. Switch to
# real pretrained embeddings with:
#   EMBED_BACKEND=openai OPENAI_API_KEY=... ./scripts/embeddings/text_action_model/train_all.sh
#
# All variables understood by per_skill/train_all_and_eval.sh are forwarded.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env \
  EMBEDDER=text \
  EMBED_SOURCE=action_model \
  EMBED_BACKEND="${EMBED_BACKEND:-hash}" \
  "${HERE}/../../per_skill/train_all_and_eval.sh" "$@"
