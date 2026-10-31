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

DRY_RUN_FLAG=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    DRY_RUN_FLAG=(--dry-run)
fi

echo "[1/4] Validating workspace sources and container build spec..."
"$PYTHON_BIN" -m deploy.modal.deploy ${DRY_RUN_FLAG[@]+"${DRY_RUN_FLAG[@]}"} validate-image

echo "[2/4] Provisioning persistent Modal Volumes..."
"$PYTHON_BIN" -m deploy.modal.deploy ${DRY_RUN_FLAG[@]+"${DRY_RUN_FLAG[@]}"} provision-storage

echo "[3/4] Ensuring pinned DeepSeek-V4.1-Flash-1017 checkpoint is downloaded..."
"$PYTHON_BIN" -m deploy.modal.deploy ${DRY_RUN_FLAG[@]+"${DRY_RUN_FLAG[@]}"} download-checkpoint

DEEPINFER_FLAG=()
if [[ "${WITH_DEEPINFER:-0}" == "1" ]]; then
    DEEPINFER_FLAG=(--with-deepinfer)
fi

echo "[4/4] Deploying 8x B200 SINGULARITY service (scale-to-zero; WITH_DEEPINFER=${WITH_DEEPINFER:-0})..."
"$PYTHON_BIN" -m deploy.modal.deploy ${DRY_RUN_FLAG[@]+"${DRY_RUN_FLAG[@]}"} deploy ${DEEPINFER_FLAG[@]+"${DEEPINFER_FLAG[@]}"}
