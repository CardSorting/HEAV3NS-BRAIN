# Post-Mortem PM-001: Modal Container Shallow-Mount Import Crashloop (`IndexError: 2`)

- **Incident Date**: 2026-10-31
- **Severity**: SEV-1 (Container failed to import entrypoint on remote Modal worker)
- **Status**: Resolved & Verified on 8× NVIDIA B200 (`im-R1TdH6JxBrlEvExHAIqaph`)
- **Affected Files**: `deploy/modal/{app.py, deepinfer_app.py, gpu_validation.py, deploy.py, image.py, health.py, benchmark.py, model_storage.py}`

---

## 1. Executive Summary

During initial deployment of `deploy/modal/app.py` to Modal, remote containers entered an immediate import crashloop before `SingularityV41Service.startup()` could execute:

```text
File "/root/app.py", line 25, in <module>
  _repo_root = str(Path(__file__).resolve().parents[2])
IndexError: 2
```

## 2. Root Cause Analysis

When `deploy/modal/app.py` is executed from the local checkout (`/Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/app.py`) or from the baked workspace path (`/workspace/singularity/deploy/modal/app.py`), `Path(__file__).resolve().parents` has at least 3 levels (`deploy/modal`, `deploy`, and the repository root at `.parents[2]`).

However, when Modal's container harness boots a function or class defined in `deploy/modal/app.py`, Modal mounts the entrypoint file directly at `/root/app.py` inside the container. For `/root/app.py`, `Path("/root/app.py").resolve().parents` contains only two elements:
- `parents[0] == Path("/root")`
- `parents[1] == Path("/")`

Indexing `.parents[2]` raised `IndexError: 2` at module import time.

## 3. Remediation

Replaced hardcoded `.parents[2]` indexing across all `deploy/modal/*.py` modules with `_resolve_repo_root(file_path: Path) -> Path`:

```python
def _resolve_repo_root(file_path: Path) -> Path:
    resolved = file_path.resolve()
    for parent in resolved.parents:
        if (parent / "deploy" / "modal").is_dir() or (parent / "Cargo.toml").is_file():
            return parent
    candidates = [
        Path("/workspace/singularity"),
        Path("/opt/singularity"),
        Path.cwd(),
        *(Path(p) for p in sys.path if p),
    ]
    for candidate in candidates:
        if (candidate / "deploy" / "modal").is_dir() or (candidate / "Cargo.toml").is_file():
            return candidate
    return resolved.parent
```

## 4. Preventative Guardrails & Regression Tests

Added `ModalPipelineTests.test_script_mode_and_modal_cli_import_refs` in [`deploy/modal/test_modal_pipeline.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/test_modal_pipeline.py#L483-L532), which compiles and executes every module in `deploy/modal/*.py` with `__file__ = "/root/<name>.py"` and `__file__ = "/app.py"` to guarantee shallow mounts never raise `IndexError`.

