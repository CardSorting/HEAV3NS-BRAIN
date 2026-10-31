# Post-Mortem PM-003: `KeyError: 'shard_count'` in `SingularityV41Service.startup()`

- **Incident Date**: 2026-10-31
- **Severity**: SEV-2 (Container startup raised `KeyError: 'shard_count'` when logging checkpoint summary)
- **Status**: Resolved & Verified on 8× NVIDIA B200 (`im-R1TdH6JxBrlEvExHAIqaph`)
- **Affected Files**: [`deploy/modal/model_storage.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/model_storage.py), [`deploy/modal/app.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/app.py)

---

## 1. Executive Summary

Following the deployment of `ensure_checkpoint_ready()` in `deploy/modal/app.py`, `SingularityV41Service.startup()` failed at container startup with:

```text
File "/root/app.py", line 496, in ensure_checkpoint_ready
  f"({check['shard_count']} shards, {check['total_bytes']} bytes)."
KeyError: 'shard_count'
```

## 2. Root Cause Analysis

`validate_checkpoint_directory()` in [`deploy/modal/model_storage.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/model_storage.py) previously returned three different dictionary schemas depending on which return path was taken:
1. Early return when directory did not exist: `{"valid": False, "errors": [...], "root": ...}` (missing `shard_count` and `total_bytes`).
2. Early return when `config.json` / `tokenizer.json` / `model.safetensors.index.json` was missing: `{"valid": False, "errors": [...], "root": ...}` (missing `shard_count` and `total_bytes`).
3. Normal return path: returned `"present_shards"` and `"total_shard_file_bytes"`, whereas callers in `app.py` accessed `check["shard_count"]` and `check["total_bytes"]`.

## 3. Remediation

1. **Standardized 15-Key Return Schema in `validate_checkpoint_directory`**:
   - Every return path (missing directory, missing metadata files, or full shard verification) now returns the exact same 15 keys:
     `valid`, `checkpoint_dir`, `index_sha256`, `referenced_shards`, `present_shards`, `shard_count`, `indexed_tensors`, `verified_header_tensors`, `verified_csf_scales`, `schema`, `codec`, `total_size_metadata`, `total_shard_file_bytes`, `total_bytes`, and `errors`.
2. **Defensive `.get()` Logging in `ensure_checkpoint_ready`**:
   - Updated `ensure_checkpoint_ready()` in `deploy/modal/app.py` to use `check.get('shard_count', check.get('present_shards', 0))` and `check.get('total_bytes', check.get('total_shard_file_bytes', 0))`.

## 4. Preventative Guardrails & Regression Tests

Updated `ModalPipelineTests.test_safetensors_header_and_checkpoint_validation` in [`deploy/modal/test_modal_pipeline.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/test_modal_pipeline.py#L225-L314) to assert `set(missing_dir.keys()) == set(non_strict.keys())` and `set(missing_meta.keys()) == set(non_strict.keys())` across all return paths.

