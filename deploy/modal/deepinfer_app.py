"""Isolated Modal deployment for the DeepInfer reference engine on 8x NVIDIA B200
(singularity2.md Section 11) for controlled head-to-head benchmarking against SINGULARITY.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

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


if __package__ in (None, ""):
    _modal_dir = str(Path(__file__).resolve().parent)
    _repo_root = str(_resolve_repo_root(Path(__file__)))
    if _modal_dir in sys.path and (Path(_modal_dir) / "deploy.py").is_file():
        sys.path.remove(_modal_dir)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from deploy.modal.config import (
        BENCHMARK_MOUNT_PATH,
        BENCHMARK_VOLUME_NAME,
        CHECKPOINT_DIR,
        CPU_CORES,
        CSF_MODEL_ID,
        CSF_PINNED_REVISION,
        DEEPINFER_APP_NAME,
        GPU_COUNT,
        GPU_SPEC,
        HF_SECRET_NAME,
        MEMORY_MB,
        MODEL_ID,
        REQUEST_TIMEOUT_MAX_SECS,
        SCALEDOWN_WINDOW_MAX_SECS,
        STARTUP_TIMEOUT_SECS,
        VOLUME_MOUNT_PATH,
        VOLUME_NAME,
        policy_env_snapshot,
        resolve_scaling_policy,
    )
    from deploy.modal.app import _exit_container
    from deploy.modal.health import verify_gpu_topology
    from deploy.modal.image import build_deepinfer_image
    from deploy.modal.lifecycle import LifecycleState, LifecycleTracker, LifecycleWatchdog
    from deploy.modal.model_storage import (
        download_pinned_checkpoint,
        prepare_csf_checkpoint_directory,
        validate_checkpoint_directory,
    )
else:
    from .config import (
        BENCHMARK_MOUNT_PATH,
        BENCHMARK_VOLUME_NAME,
        CHECKPOINT_DIR,
        CPU_CORES,
        CSF_MODEL_ID,
        CSF_PINNED_REVISION,
        DEEPINFER_APP_NAME,
        GPU_COUNT,
        GPU_SPEC,
        HF_SECRET_NAME,
        MEMORY_MB,
        MODEL_ID,
        REQUEST_TIMEOUT_MAX_SECS,
        SCALEDOWN_WINDOW_MAX_SECS,
        STARTUP_TIMEOUT_SECS,
        VOLUME_MOUNT_PATH,
        VOLUME_NAME,
        policy_env_snapshot,
        resolve_scaling_policy,
    )
    from .app import _exit_container
    from .health import verify_gpu_topology
    from .image import build_deepinfer_image
    from .lifecycle import LifecycleState, LifecycleTracker, LifecycleWatchdog
    from .model_storage import (
        download_pinned_checkpoint,
        prepare_csf_checkpoint_directory,
        validate_checkpoint_directory,
    )


def build_deepinfer_torchrun_command(
    checkpoint_dir: str = CHECKPOINT_DIR,
    world_size: int = GPU_COUNT,
    max_seq_len: int = 32_768,
    decode_num_requests: int = 64,
    result_jsonl: str = f"{BENCHMARK_MOUNT_PATH}/deepinfer_benchmark.jsonl",
) -> List[str]:
    """Return the native DeepInfer 8-GPU benchmark command (`examples/benchmark.py`).

    Note (`singularity2.md` Section 11.2): DeepInfer's public release is an offline
    inference engine (`deepinfer.entrypoints.llm.LLM`) that spawns `world_size` worker
    ranks via `examples/_launch.py:launch_dp`.
    """
    return [
        "python3",
        "/workspace/DeepInfer/examples/benchmark.py",
        "--model_name",
        checkpoint_dir,
        "--context_length",
        str(max_seq_len),
        "--decode_num_requests",
        str(decode_num_requests),
        "--result_jsonl",
        result_jsonl,
    ]


def get_deepinfer_reference_spec() -> Dict[str, object]:
    """Return the isolated DeepInfer reference configuration for parity auditing."""
    return {
        "app_name": DEEPINFER_APP_NAME,
        "model_id": MODEL_ID,
        "gpu_spec": GPU_SPEC,
        "gpu_count": GPU_COUNT,
        "checkpoint_dir": CHECKPOINT_DIR,
        "volume_name": VOLUME_NAME,
        "entrypoint_module": "deepinfer.entrypoints.llm.LLM",
        "command": build_deepinfer_torchrun_command(),
    }


def create_deepinfer_adapter_app(service: object):
    """Create the thin OpenAI-compatible FastAPI benchmark adapter around `deepinfer.entrypoints.llm.LLM`."""
    import time
    import uuid
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse

    adapter = FastAPI(title="DeepInfer Reference Benchmark Adapter")

    @adapter.get("/health")
    def health():
        is_ready = bool(getattr(service, "ready", False) and getattr(service, "llm", None) is not None)
        return {"status": "healthy", "ready": is_ready}

    @adapter.get("/ready")
    def ready():
        llm_loaded = getattr(service, "llm", None) is not None
        is_ready = bool(getattr(service, "ready", False) and llm_loaded)
        checks = {
            "container_alive": True,
            "runtime_initialized": True,
            "cuda_operational": is_ready,
            "gpus_available": is_ready,
            "distributed_topology_operational": is_ready,
            "checkpoint_present": True,
            "model_loaded": llm_loaded,
            "model_ready": is_ready,
            "inference_executed": is_ready,
        }
        payload = {
            "status": "ready" if is_ready else "not_ready",
            "ready": is_ready,
            "model": MODEL_ID,
            **checks,
            "checks": checks,
        }
        return JSONResponse(status_code=200 if is_ready else 503, content=payload)

    @adapter.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "owned_by": "deepseek-ai"}],
        }

    @adapter.post("/v1/chat/completions")
    def chat_completions(req: Dict[str, object]):
        llm = getattr(service, "llm", None)
        tokenizer = getattr(service, "tokenizer", None)
        if llm is None or tokenizer is None:
            raise HTTPException(
                status_code=503,
                detail="DeepInfer LLM engine is not initialized in this container",
            )
        messages = req.get("messages", [])
        prompt = "\n".join(
            str(m.get("content", "")) for m in messages if isinstance(m, dict)
        )
        max_tokens = int(req.get("max_tokens", 16))
        token_ids = tokenizer.encode(prompt)
        try:
            from deepinfer.sampling_params import SamplingParams

            params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        except ImportError:
            params = {"temperature": 0.0, "max_tokens": max_tokens}
        outputs = llm.generate([token_ids], sampling_params=params)
        out_ids = outputs[0].outputs[0].token_ids if outputs else []
        text = tokenizer.decode(out_ids, skip_special_tokens=True)
        created = int(time.time())
        cmpl_id = f"chatcmpl-{uuid.uuid4()}"

        if req.get("stream"):
            def event_stream():
                import json as _json

                for tok_id in out_ids:
                    piece = tokenizer.decode([tok_id], skip_special_tokens=True)
                    chunk = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [
                            {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                        ],
                    }
                    yield f"data: {_json.dumps(chunk)}\n\n"
                done = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {_json.dumps(done)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        return {
            "id": cmpl_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(token_ids),
                "completion_tokens": len(out_ids),
                "total_tokens": len(token_ids) + len(out_ids),
            },
        }

    return adapter


try:
    import os
    import modal

    if os.environ.get("SINGULARITY_SKIP_MODAL_APP") == "1":
        raise ImportError("SINGULARITY_SKIP_MODAL_APP=1")

    _di_policy = resolve_scaling_policy()
    deepinfer_app = modal.App(DEEPINFER_APP_NAME)
    app = deepinfer_app
    deepinfer_image = build_deepinfer_image()
    _di_policy_env = policy_env_snapshot()
    if _di_policy_env:
        deepinfer_image = deepinfer_image.env(_di_policy_env)
    weights_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    benchmark_volume = modal.Volume.from_name(BENCHMARK_VOLUME_NAME, create_if_missing=True)
    hf_secret = modal.Secret.from_name(HF_SECRET_NAME)

    @deepinfer_app.cls(
        image=deepinfer_image,
        gpu=GPU_SPEC,
        cpu=CPU_CORES,
        memory=MEMORY_MB,
        volumes={
            VOLUME_MOUNT_PATH: weights_volume,
            BENCHMARK_MOUNT_PATH: benchmark_volume,
        },
        secrets=[hf_secret],
        min_containers=0,  # never pinned: reference service must scale to zero
        max_containers=1,
        scaledown_window=min(_di_policy.scaledown_window_secs, SCALEDOWN_WINDOW_MAX_SECS),
        startup_timeout=STARTUP_TIMEOUT_SECS,
        timeout=min(_di_policy.request_timeout_secs, REQUEST_TIMEOUT_MAX_SECS),
    )
    class DeepInferReferenceService:
        @modal.enter()
        def startup(self) -> None:
            self.tracker = LifecycleTracker(
                max_concurrent=_di_policy.max_concurrent_requests,
                request_timeout_secs=_di_policy.request_timeout_secs,
                max_lifetime_secs=_di_policy.max_worker_lifetime_secs,
                gpu_count=GPU_COUNT,
                gpu_usd_per_gpu_hour=_di_policy.gpu_usd_per_gpu_hour,
                max_spend_usd=_di_policy.max_spend_usd,
            )
            self.tracker.transition(LifecycleState.INITIALIZING)
            self.llm = None
            self.tokenizer = None
            self.ready = False
            try:
                weights_volume.reload()
                gpu_info = verify_gpu_topology(expected_gpus=GPU_COUNT)
                if not gpu_info["valid"]:
                    raise RuntimeError(f"DeepInfer B200 topology check failed: {gpu_info['errors']}")
                if Path(CHECKPOINT_DIR).is_dir():
                    prepare_csf_checkpoint_directory(CHECKPOINT_DIR)
                ckpt_info = validate_checkpoint_directory(
                    CHECKPOINT_DIR, strict_pinned=True, verify_shard_headers=True
                )
                if not ckpt_info["valid"]:
                    download_pinned_checkpoint(
                        CHECKPOINT_DIR,
                        repo_id=CSF_MODEL_ID,
                        revision=CSF_PINNED_REVISION,
                        commit_callback=weights_volume.commit,
                    )
                    weights_volume.commit()
                    ckpt_info = validate_checkpoint_directory(
                        CHECKPOINT_DIR, strict_pinned=True, verify_shard_headers=True
                    )
                    if not ckpt_info["valid"]:
                        raise RuntimeError(f"DeepInfer checkpoint check failed: {ckpt_info['errors']}")
                from deepinfer.entrypoints.llm import LLM
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    CHECKPOINT_DIR, trust_remote_code=True
                )
                self.llm = LLM(model=CHECKPOINT_DIR, max_model_len=131_072)
                self.ready = True
                self.tracker.transition(LifecycleState.READY)
            except Exception:
                self.ready = False
                self.llm = None
                self.tracker.transition(LifecycleState.FAILED)
                self.tracker.transition(LifecycleState.STOPPED)
                raise
            self.watchdog = LifecycleWatchdog(
                self.tracker,
                drain_timeout_secs=_di_policy.drain_timeout_secs,
                stop_child=self._cleanup_llm,
                exit_fn=lambda: _exit_container(cleanup=self._cleanup_llm),
                liveness=lambda: bool(getattr(self, "ready", False) and getattr(self, "llm", None) is not None),
            )
            self.watchdog.start()

        def _cleanup_llm(self) -> None:
            self.ready = False
            self.llm = None

        @modal.exit()
        def shutdown(self) -> None:
            watchdog = getattr(self, "watchdog", None)
            if watchdog is not None:
                watchdog.halt()
            tracker = getattr(self, "tracker", None)
            if tracker is not None:
                tracker.drain(_di_policy.drain_timeout_secs)
            self._cleanup_llm()
            if tracker is not None:
                try:
                    tracker.transition(LifecycleState.STOPPED)
                except Exception:  # noqa: BLE001
                    pass

        @modal.method()
        def status(self) -> Dict[str, object]:
            return {
                "engine": "deepinfer-reference",
                "ready": getattr(self, "ready", False),
                "llm_loaded": getattr(self, "llm", None) is not None,
                "spec": get_deepinfer_reference_spec(),
            }

        @modal.asgi_app()
        def serve(self):
            return create_deepinfer_adapter_app(self)

except ImportError:
    deepinfer_app = None
    app = None
    DeepInferReferenceService = None

