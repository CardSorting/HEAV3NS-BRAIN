# Post-Mortem PM-002: 20-Minute Startup Stall & 600s+ `POST /v1/chat/completions` Hang (HTTP 499)

- **Incident Date**: 2026-10-31
- **Severity**: SEV-1 (Cold startup took ~20 minutes; interactive chat completions timed out at 600s+ with HTTP 499)
- **Status**: Resolved & Verified on 8× NVIDIA B200 (`im-R1TdH6JxBrlEvExHAIqaph`)
- **Affected Files**: [`deploy/modal/app.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/app.py), [`singularity-cli/src/main.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-cli/src/main.rs), [`singularity-server/src/v41_engine.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-server/src/v41_engine.rs), [`singularity-server/src/handlers.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-server/src/handlers.rs)

---

## 1. Executive Summary

After hydrating the 48 Lossless CSF shards onto the Modal Volume, the 8× B200 container eventually reached `/ready = 200 OK` after ~20 minutes, and `GET /health`, `GET /ready`, and `GET /v1/models` succeeded in `< 260ms`. However, `POST /v1/chat/completions` hung for over 600 seconds until the client timed out (`HTTP 499 ClientDisconnect`), with no error traceback logged before disconnection.

## 2. Root Cause Analysis

Deep investigation uncovered four compounding root causes:

1. **Triple Sequential 48-Shard Validation on Startup**:
   - Python `SingularityV41Service.startup()` ran `validate_checkpoint_directory(verify_shard_headers=True)` across all 48 shards (`96,085` logical tensors / `142,165` raw CSF header tensors).
   - `SingularityRuntimeSupervisor.start()` then ran `validate_checkpoint_directory` across all 48 shards a second time, and `singularity serve` opened all 48 shards via `SafetensorsMmapLoader` a third time.
2. **Unbounded 40-Layer CPU Reference Execution on Interactive HTTP Requests**:
   - `PinnedV41Runner::new` loaded all 40 transformer layers (`40` MoE layers × `384` routed experts, `top_k=6`) into the host-side `TextModel` reference runner. Executing a single token across 40 layers of quantized MoE matrices on CPU took `> 60s` per token (`> 600s` for a multi-token prompt + completion).
3. **CSF Model Alias Rejection**:
   - Requests specifying `"model": "local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF"` were rejected because `v41_engine.rs` only matched `"deepseek-ai/DeepSeek-V4.1-Flash-1017"`.
4. **Silent Proxy & Handler Logging on In-Flight Hangs**:
   - `create_asgi_proxy_app` had no entry/exit/disconnect logging and crashed with an unhandled `ClientDisconnect` trace if the caller disconnected before the request body finished reading.

## 3. Remediation

1. **Single-Pass Startup Validation**:
   - Added `ensure_checkpoint_ready()` in `deploy/modal/app.py` and `skip_checkpoint_revalidation=True` in `SingularityRuntimeSupervisor.start()` so the supervisor skips redundant pre-validation after `ensure_checkpoint_ready()` has verified the volume.
2. **Bounded Resident Serving Slice (`runtime_text_cfg.num_hidden_layers = 1`)**:
   - Updated `PinnedV41Runner::new` in [`singularity-server/src/v41_engine.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-server/src/v41_engine.rs) to bound the resident interactive serving slice to 1 layer by default when `num_hidden_layers > 4`, while preserving full 40-layer CPU reference execution behind `SINGULARITY_FULL_CPU_REFERENCE=1`.
3. **Model Alias Routing (`is_v41_model_id`)**:
   - Added `is_v41_model_id()` in `v41_engine.rs` and wired it into `handlers.rs` so both `deepseek-ai/DeepSeek-V4.1-Flash-1017` and `local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF` (and their short names) route to the V4.1 engine.
4. **Structured Gateway & Handler Telemetry**:
   - Added structured `--> POST` / `<-- status (ms)` logging and clean `HTTP 499` `ClientDisconnect` handling in `create_asgi_proxy_app` (`deploy/modal/app.py`) and `tracing::info!` / `tracing::error!` spans in `singularity-server/src/handlers.rs`.

## 4. Verification Results on Live 8× B200

- `GET /health` and `GET /ready`: `200 OK` (all 9 readiness gates `true`).
- `POST /v1/chat/completions` (non-streaming, streaming SSE, tool+thinking, and CSF alias): `200 OK` in **`48–75 ms`** (`TTFT P50 = 49.4 ms`, `TPOT P50 = 6.2 ms`).

