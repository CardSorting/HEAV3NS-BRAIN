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
DEEPINFER_URL="${2:-${DEEPINFER_ENDPOINT_URL:-}}"
OUTPUT_PATH="${BENCHMARK_OUTPUT_PATH:-./benchmark_report.json}"
DRY_RUN_FLAG=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    DRY_RUN_FLAG=(--dry-run)
fi

DEEPINFER_ARGS=()
if [[ -n "${DEEPINFER_URL}" ]]; then
    DEEPINFER_ARGS=(--deepinfer-url "${DEEPINFER_URL}")
fi

echo "Running controlled SINGULARITY benchmark suite against ${ENDPOINT_URL}..."
"$PYTHON_BIN" -m deploy.modal.deploy ${DRY_RUN_FLAG[@]+"${DRY_RUN_FLAG[@]}"} benchmark \
    --url "${ENDPOINT_URL}" \
    ${DEEPINFER_ARGS[@]+"${DEEPINFER_ARGS[@]}"} \
    --output "${OUTPUT_PATH}"
