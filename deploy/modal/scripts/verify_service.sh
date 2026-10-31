#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${HOME}/.pyenv/versions/3.13.6/bin/python" ]]; then
        PYTHON_BIN="${HOME}/.pyenv/versions/3.13.6/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

ENDPOINT_URL="${1:-${SINGULARITY_ENDPOINT_URL:-http://127.0.0.1:8080}}"
DRY_RUN_FLAG=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    DRY_RUN_FLAG=(--dry-run)
fi

echo "[1/3] Inspecting deployment configuration and 17-step startup contract..."
"$PYTHON_BIN" -m deploy.modal.deploy ${DRY_RUN_FLAG[@]+"${DRY_RUN_FLAG[@]}"} inspect

echo "[2/3] Checking /health, /ready (9 operational flags), and /v1/models on ${ENDPOINT_URL}..."
"$PYTHON_BIN" -m deploy.modal.deploy ${DRY_RUN_FLAG[@]+"${DRY_RUN_FLAG[@]}"} health --url "${ENDPOINT_URL}"

echo "[3/3] Executing end-to-end smoke inference (chat, thinking, tool calls, SSE)..."
"$PYTHON_BIN" -m deploy.modal.deploy ${DRY_RUN_FLAG[@]+"${DRY_RUN_FLAG[@]}"} test-inference --url "${ENDPOINT_URL}"
