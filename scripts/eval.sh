#!/bin/bash
# Запуск agent_eval harness.
# Использование: bash scripts/eval.sh [user_id]
set -e
USER_ID="${1:-eval_$(date +%s)}"
echo "Running eval for user: $USER_ID"
python -m GCN.eval.run_eval "$USER_ID"
