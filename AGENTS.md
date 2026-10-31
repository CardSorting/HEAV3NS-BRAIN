# AGENTS.md — High-Throughput Agent & Developer Execution Guide

This document defines the workspace map, verification gates, scale-to-zero cost-governance rules, and high-throughput execution ergonomics for AI coding agents and human developers working in `HEAV3NS-BRAIN` (Project SINGULARITY).

---

## 1. Fast Navigation & Component Map

| Path | Responsibility |
|---|---|
| [`Makefile`](Makefile) | Unified entrypoints (`make check`, `make test`, `make lint`, `make audit`, `make teardown`) |
| [`singularity-core/`](singularity-core/) | Core types (`Sequence`, `SingularityError`, `SystemConfig`, `MetricsCollector`, `TokenBlock`) |
| [`singularity-cuda/`](singularity-cuda/) | Hardware detection, `GpuBackend` trait, `MockGpuBackend` (SM100/SM90), cluster topology planner |
| [`singularity-cache/`](singularity-cache/) | `PagedBlockManager` (bitset + ref-counted KV blocks), `RadixTree` prefix cache, FP4/MXFP8 layouts |
| [`singularity-scheduler/`](singularity-scheduler/) | `ZosScheduler` (Zero-Overhead Scheduler), `LcfsScheduler`, `DpConsensusEngine`, `RankHealthMonitor`, `ExternalStepGuard` |
| [`singularity-model/`](singularity-model/) | Safetensors mmap loader, `V41Runtime` (DeepSeek-V4.1-Flash-1017), MLA, MoE, Engram, `model_error` |
| [`singularity-grammar/`](singularity-grammar/) | Streaming KMP matcher and JSON Schema bitmask constrained decoding |
| [`singularity-server/`](singularity-server/) | Axum OpenAI-compatible HTTP/SSE server, `CircuitBreaker`, `LeakyTokenBucket`, `StreamRingBuffer`, `spawn_rank_watchdog` |
| [`singularity-cli/`](singularity-cli/) | `singularity` binary (`serve`, `verify-checkpoint`, `benchmark`, `inspect-hardware`, `validate-config`) |
| [`singularity-tests/`](singularity-tests/) | End-to-end HTTP server tests (`e2e_server.rs`), rank-health isolation tests (`rank_health_serve.rs`), failure-mode suites |
| [`deploy/modal/`](deploy/modal/) | Scale-to-zero Modal infrastructure: [`config.py`](deploy/modal/config.py), [`lifecycle.py`](deploy/modal/lifecycle.py), [`ops.py`](deploy/modal/ops.py), [`app.py`](deploy/modal/app.py), [`deepinfer_app.py`](deploy/modal/deepinfer_app.py), [`smoke_app.py`](deploy/modal/smoke_app.py), [`deploy.py`](deploy/modal/deploy.py) |
| [`docs/SCALE_TO_ZERO_REPORT.md`](docs/SCALE_TO_ZERO_REPORT.md) | Authoritative verification report and requirement-to-evidence matrix for `singularity3.md` |

---

## 2. High-Throughput Parent-Agent Flow & Blocker Prevention

To maximize execution velocity and prevent sandbox or orchestration stalls:

1. **Direct Parent Execution vs. Subagent Delegation**:
   - **Run build, test, lint, and CLI verification commands directly in the parent agent** (`run_command`). Subagents run in a restricted sandbox where unsandboxed shell commands (`cargo`, `.venv/bin/python`, `modal`) may be denied by default system policy unless explicitly allowlisted.
   - Use read-only `research` subagents only for broad, multi-file exploration when context conservation is needed; do not delegate command-heavy verification or tight edit-compile-test loops to subagents unless command allowlists are active.
2. **Allowlisting Commands for Zero-Prompt Velocity**:
   - In Antigravity IDE (`Settings > Permissions` or `/permissions`), add workspace allowlist rules for:
     - `cargo test*`, `cargo check*`, `cargo clippy*`, `cargo fmt*`, `cargo doc*`
     - `.venv/bin/python -W ignore -m unittest*`
     - `ruff check*`
     - `modal container list*`, `modal app list*`
3. **Interactive Decisions under `/goal` Mode**:
   - When `/goal` mode is active, the stop hook blocks plain-text questions until the goal is complete. Whenever a genuine user decision is required (such as approving paid 8×B200 GPU spend), **always use the `ask_question` tool** rather than ending a turn with a plain-text question.
4. **Local Python & CLI Interpreters**:
   - **Unit tests & FastAPI gateway tests**: Use `.venv/bin/python` (which has `fastapi` and `httpx` installed so all 71 tests in `deploy/modal/` execute with 0 skipped tests):
     ```bash
     .venv/bin/python -W ignore -m unittest discover -s deploy/modal -p "test_*.py"
     .venv/bin/python -W ignore -m unittest discover -s scripts -p "test_*.py"
     ```
   - **Modal CLI**: Use `modal` (installed at `/Users/bozoegg/.pyenv/versions/3.13.6/bin/modal`).
   - **Standalone Script-Mode Compatibility**: Every executable module in `deploy/modal/*.py` (`app.py`, `deepinfer_app.py`, `smoke_app.py`, `deploy.py`, `ops.py`, `validate_checkpoint.py`) intentionally defines `_resolve_repo_root()` at the top of the file so it can be executed both as `python3 -m deploy.modal.<name>` and inside a Modal container as `/root/<name>.py`. This is enforced by `test_script_mode_and_modal_cli_import_refs` in [`deploy/modal/test_modal_pipeline.py`](deploy/modal/test_modal_pipeline.py).

---

## 3. Scale-to-Zero & Cost-Governance Invariants

1. **Zero Idle Compute by Default**:
   - `min_containers` is `0` by default across all Modal apps (`singularity-v41-flash-1017`, `singularity-deepinfer-v41`, `singularity-smoke`).
   - Setting `SINGULARITY_MIN_CONTAINERS > 0` is rejected unless `SINGULARITY_ALLOW_PINNED_WARM=1` is also set.
2. **Deploy-Time Policy Snapshot**:
   - `config.policy_env_snapshot()` bakes whitelisted `SINGULARITY_*` scaling/guard variables into the Modal image so in-container `LifecycleTracker` and `LifecycleWatchdog` enforce the exact operator-configured bounds.
3. **No Paid 8×B200 Runs Without Explicit Approval**:
   - Never deploy or trigger requests against `singularity-v41-flash-1017` or `singularity-deepinfer-v41` (8×B200) without explicit user confirmation.
   - Always verify zero running containers after any cloud operation:
     ```bash
     make audit
     modal container list --json   # must be []
     ```
   - Emergency full teardown:
     ```bash
     make teardown
     ```

---

## 4. Pre-Commit Verification Checklist

Run `make check` (or the individual commands below) before concluding any engineering pass:

```bash
cargo fmt --all -- --check
cargo clippy --all --offline -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --all --no-deps --offline
cargo test --workspace --offline
ruff check deploy/modal scripts
.venv/bin/python -W ignore -m unittest discover -s deploy/modal -p "test_*.py"
.venv/bin/python -W ignore -m unittest discover -s scripts -p "test_*.py"
```

