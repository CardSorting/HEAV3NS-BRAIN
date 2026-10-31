"""Hardware, readiness, and end-to-end inference health verifiers for SINGULARITY on Modal."""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    from deploy.modal.config import CSF_MODEL_ID, EXPECTED_SM_MAJOR, EXPECTED_SM_MINOR, GPU_COUNT, MODEL_ID
else:
    from .config import CSF_MODEL_ID, EXPECTED_SM_MAJOR, EXPECTED_SM_MINOR, GPU_COUNT, MODEL_ID

REQUIRED_READINESS_KEYS: List[str] = [
    "container_alive",
    "runtime_initialized",
    "cuda_operational",
    "gpus_available",
    "distributed_topology_operational",
    "checkpoint_present",
    "model_loaded",
    "model_ready",
    "inference_executed",
]


def verify_gpu_topology(
    expected_gpus: int = GPU_COUNT,
    expected_sm: Tuple[int, int] = (EXPECTED_SM_MAJOR, EXPECTED_SM_MINOR),
) -> Dict[str, object]:
    """Query `nvidia-smi` inside the container to verify 8x NVIDIA B200 GPUs and SM 10.0."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,compute_cap,memory.total,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return {
            "valid": False,
            "gpu_count": 0,
            "devices": [],
            "errors": [f"nvidia-smi query failed: {exc}"],
        }

    devices: List[Dict[str, object]] = []
    errors: List[str] = []
    for raw_line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in raw_line.split(",")]
        if len(parts) < 5:
            continue
        idx_str, name, cc_str, mem_mb_str, uuid = parts[:5]
        try:
            cc_parts = [int(x) for x in cc_str.split(".")]
            sm = (cc_parts[0], cc_parts[1] if len(cc_parts) > 1 else 0)
        except ValueError:
            sm = (0, 0)
        try:
            mem_mb = int(float(mem_mb_str))
        except ValueError:
            mem_mb = 0
        devices.append(
            {
                "index": int(idx_str) if idx_str.isdigit() else len(devices),
                "name": name,
                "compute_capability": f"{sm[0]}.{sm[1]}",
                "memory_mb": mem_mb,
                "uuid": uuid,
            }
        )
        if sm[0] < expected_sm[0] or (sm[0] == expected_sm[0] and sm[1] < expected_sm[1]):
            errors.append(
                f"GPU {idx_str} ({name}) has SM {sm[0]}.{sm[1]}, expected >= {expected_sm[0]}.{expected_sm[1]}"
            )
        if "B200" not in name.upper():
            errors.append(f"GPU {idx_str} model '{name}' is not an NVIDIA B200")

    if len(devices) != expected_gpus:
        errors.append(f"Detected {len(devices)} GPUs, expected {expected_gpus}")

    return {
        "valid": len(errors) == 0,
        "gpu_count": len(devices),
        "devices": devices,
        "errors": errors,
    }


def _http_json(
    url: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, object]] = None,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> Tuple[int, Dict[str, object]]:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = {"raw_error": body}
        return exc.code, parsed


def check_service_health(
    base_url: str,
    *,
    api_key: Optional[str] = None,
    require_v41_model: bool = True,
    timeout: float = 15.0,
) -> Dict[str, object]:
    """Verify `/health`, `/ready` (including all 9 readiness flags), and `/v1/models`."""
    base = base_url.rstrip("/")
    errors: List[str] = []

    try:
        health_status, health_body = _http_json(
            f"{base}/health", api_key=api_key, timeout=timeout
        )
    except Exception as exc:
        return {
            "healthy": False,
            "ready": False,
            "errors": [f"GET /health failed: {exc}"],
        }

    if health_status != 200 or health_body.get("status") != "healthy":
        errors.append(f"/health returned HTTP {health_status}: {health_body}")

    try:
        ready_status, ready_body = _http_json(
            f"{base}/ready", api_key=api_key, timeout=timeout
        )
    except Exception as exc:
        return {
            "healthy": health_status == 200,
            "ready": False,
            "health": health_body,
            "errors": errors + [f"GET /ready failed: {exc}"],
        }

    checks = ready_body.get("checks", {})
    if not isinstance(checks, dict):
        checks = {}
    for key in REQUIRED_READINESS_KEYS:
        val = checks.get(key, ready_body.get(key))
        if val is not True:
            errors.append(f"Readiness check '{key}' is not True (got {val!r})")

    if ready_status != 200 or ready_body.get("ready") is not True:
        errors.append(f"/ready returned HTTP {ready_status} (ready={ready_body.get('ready')})")

    models_list: List[str] = []
    try:
        models_status, models_body = _http_json(
            f"{base}/v1/models", api_key=api_key, timeout=timeout
        )
        if models_status != 200:
            errors.append(f"/v1/models returned HTTP {models_status}")
        else:
            for item in models_body.get("data", []):
                if isinstance(item, dict) and "id" in item:
                    models_list.append(str(item["id"]))
            if require_v41_model and MODEL_ID not in models_list:
                errors.append(f"Required model '{MODEL_ID}' not listed in /v1/models: {models_list}")
    except Exception as exc:
        errors.append(f"GET /v1/models failed: {exc}")

    return {
        "healthy": health_status == 200,
        "ready": len(errors) == 0,
        "health": health_body,
        "readiness": ready_body,
        "models": models_list,
        "errors": errors,
    }


def run_smoke_inference(
    base_url: str,
    *,
    model: str = MODEL_ID,
    api_key: Optional[str] = None,
    timeout: float = 600.0,
) -> Dict[str, object]:
    """Execute end-to-end smoke inference tests (chat, thinking mode, tool definitions, and SSE stream)."""
    base = base_url.rstrip("/")
    results: Dict[str, object] = {}
    errors: List[str] = []

    # 1. Standard non-streaming chat completion
    chat_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Say hello in three words."},
        ],
        "max_tokens": 4,
        "stream": False,
    }
    try:
        status, body = _http_json(
            f"{base}/v1/chat/completions",
            method="POST",
            payload=chat_payload,
            api_key=api_key,
            timeout=timeout,
        )
    except Exception as exc:
        status, body = 599, {"error": str(exc)}
    if status != 200:
        errors.append(f"Chat completion returned HTTP {status}: {body}")
    results["chat"] = {"status": status, "response": body}

    # 2. Tool-enabled chat completion
    if model in (
        MODEL_ID,
        CSF_MODEL_ID,
        "DeepSeek-V4.1-Flash-1017",
        "DeepSeek-V4.1-Flash-1017-lossless-CSF",
    ):
        tool_payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "What is the weather in Tokyo?"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "thinking": "thinking",
            "reasoning_effort": 50,
            "max_tokens": 8,
            "stream": False,
        }
        try:
            t_status, t_body = _http_json(
                f"{base}/v1/chat/completions",
                method="POST",
                payload=tool_payload,
                api_key=api_key,
                timeout=timeout,
            )
        except Exception as exc:
            t_status, t_body = 599, {"error": str(exc)}
        if t_status != 200:
            errors.append(f"Tool/thinking chat completion returned HTTP {t_status}: {t_body}")
        results["tool_thinking_chat"] = {"status": t_status, "response": t_body}

    # 3. Streaming SSE chat completion
    stream_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Count to 3."}],
        "max_tokens": 4,
        "stream": True,
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(stream_payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            sse_body = resp.read().decode("utf-8")
            done_seen = "[DONE]" in sse_body
            if not done_seen:
                errors.append("SSE stream did not terminate with [DONE]")
            results["sse_stream"] = {
                "status": resp.status,
                "done_received": done_seen,
                "bytes": len(sse_body),
            }
    except Exception as exc:
        errors.append(f"SSE streaming request failed: {exc}")

    return {
        "passed": len(errors) == 0,
        "model": model,
        "results": results,
        "errors": errors,
    }
