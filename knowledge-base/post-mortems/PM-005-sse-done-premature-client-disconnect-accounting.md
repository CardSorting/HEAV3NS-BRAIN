# Post-Mortem PM-005: False SSE Stream Abandonment & Missing Token Accounting on Client Disconnect After `data: [DONE]`

- **Incident Date**: 2026-10-31
- **Severity**: SEV-3 (Completed OpenAI SSE streams were misclassified as `failed` / `"stream abandoned before completion"` with `tokens_out = 0` and triggered `GeneratorExit` cleanup warnings)
- **Status**: Resolved & Locked by Automated Gateway Tests
- **Affected Files**: [`deploy/modal/app.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/app.py), [`deploy/modal/test_lifecycle.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/test_lifecycle.py)

---

## 1. Executive Summary

During live cold-start measurement (`deploy.py measure-cold-start`) against `singularity-smoke` on `2026-10-31`, container logs showed:

```text
[SINGULARITY proxy] <-- POST /v1/chat/completions stream abandoned before completion (14.2ms); slot released
RuntimeError: cannot await resp.aclose() inside GeneratorExit
```

even though the client (`ops.measure_cold_start`) received every SSE token chunk and `data: [DONE]` with `HTTP 200`.

---

## 2. Root Cause Analysis

1. **Standard OpenAI SSE Client Behavior**:
   - Standard OpenAI SSE consumers (including `ops.measure_cold_start` using `urllib.request` and the official OpenAI Python/TypeScript SDKs) read lines from the HTTP response stream until they encounter `data: [DONE]`, at which point they immediately break out of their read loop and close the socket.
2. **Starlette `StreamingResponse` Generator Suspension**:
   - Inside `create_asgi_proxy_app` (`deploy/modal/app.py`), `stream_chunks()` yielded the chunk containing `data: [DONE]` and suspended at that `yield` expression.
   - Because the client closed the connection upon receiving `data: [DONE]`, Starlette exited `super().__call__(scope, receive, send)` **before** `stream_chunks()` was resumed to execute the code after the `for` loop (`completed_normally = True`).
3. **`_on_finish()` Ordering & `GeneratorExit` in `finally`**:
   - `_ReleasingStreamingResponse.__call__` called `await self._on_finish()` in its `finally` block *before* closing `self.body_iterator`. Thus `_on_finish()` saw `completed_normally == False` and recorded the request as `ok=False` (`failed += 1`, `"stream abandoned before completion"`).
   - Furthermore, when `self.body_iterator` was later garbage-collected or closed via `GeneratorExit`, the `finally:` block inside `stream_chunks()` attempted `await resp.aclose()`, which raises `RuntimeError` in Python when an async generator is being finalized via `GeneratorExit`.
4. **Missing `tokens_out` Accounting**:
   - `tracker.release(lease.request_id, ok=...)` was called without passing `tokens_out`, leaving `tokens_out` at `0` in `/__lifecycle/status` for both SSE and non-streaming JSON responses.

---

## 3. Remediation

1. **Explicit `body_iterator.aclose()` Before `_on_finish()` in [`_ReleasingStreamingResponse.__call__`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/app.py#L469-L483)**:
   - `_ReleasingStreamingResponse.__call__` now explicitly awaits `self.body_iterator.aclose()` inside its `finally` block *before* awaiting `self._on_finish()`, guaranteeing deterministic generator teardown before finalization runs.
2. **Track `saw_done` and Token Deltas in [`stream_chunks()`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/app.py#L618-L693)**:
   - `stream_chunks()` inspects SSE `data:` lines as chunks are yielded:
     - Increments `stream_state["tokens_out"]` on each SSE JSON chunk with a non-empty `choices[].delta.content` or `reasoning_content`.
     - Sets `stream_state["saw_done"] = True` as soon as `data: [DONE]` is yielded.
   - `_finalize_stream()` (which runs in normal async context inside `_on_finish()`) awaits `resp.aclose()` cleanly and treats `completed_normally or saw_done` as `ok=True` (`completed += 1`), recording `tokens_out=stream_state["tokens_out"]`.
3. **Non-Streaming JSON `tokens_out` Accounting**:
   - Non-streaming JSON responses parse `usage.completion_tokens` and pass `tokens_out` to `tracker.release(...)`.

---

## 4. Preventative Guardrails & Regression Tests

Added `test_sse_client_disconnect_after_done_counts_completed_and_tokens` and `test_non_streaming_json_records_completion_tokens` in [`deploy/modal/test_lifecycle.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/test_lifecycle.py#L616-L678) to verify both `completed == 1`, `failed == 0`, and exact `tokens_out` accounting.

