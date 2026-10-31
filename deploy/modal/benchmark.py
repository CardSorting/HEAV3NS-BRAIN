"""Controlled benchmark suite for SINGULARITY vs DeepInfer on 8x NVIDIA B200
(singularity2.md Section 12).

Enforces the Section 12.1 Functional & Correctness Gate before recording performance
metrics across Configurations A (Base), B (Optimized), C (Serving), and D (Concurrency Scaling).
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
    from deploy.modal.config import GPU_SPEC, MODEL_ID, PINNED_REVISION
else:
    from .config import GPU_SPEC, MODEL_ID, PINNED_REVISION

CONTEXT_TIERS: Dict[str, int] = {
    "short_128": 128,
    "medium_4k": 4096,
    "long_32k": 32768,
    "max_128k": 131072,
}

CONCURRENCY_LEVELS: List[int] = [1, 2, 4, 8, 16, 32, 64]


def percentile(values: List[float], pct: float) -> float:
    """Compute linear-interpolated percentile (0..100) on a non-empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


@dataclass
class RequestSample:
    ttft_ms: float
    total_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    itl_ms_list: List[float]
    finish_reason: str

    @property
    def tpot_ms(self) -> float:
        if self.completion_tokens <= 1:
            return 0.0
        return (self.total_latency_ms - self.ttft_ms) / max(1, self.completion_tokens - 1)


def summarize_samples(
    samples: List[RequestSample],
    wall_time_secs: float,
    configuration: str,
    context_tier: str,
    concurrency: int,
) -> Dict[str, object]:
    """Compute P50/P95/P99 latency, TTFT, ITL, TPOT, aggregate throughput, and Optimal Runtime Experience (ORE) balance metrics."""
    ttfts = [s.ttft_ms for s in samples]
    latencies = [s.total_latency_ms for s in samples]
    tpots = [s.tpot_ms for s in samples if s.completion_tokens > 1]
    all_itls = [itl for s in samples for itl in s.itl_ms_list]
    total_prompt = sum(s.prompt_tokens for s in samples)
    total_completion = sum(s.completion_tokens for s in samples)
    safe_wall = max(wall_time_secs, 1e-6)

    ttft_p50 = percentile(ttfts, 50)
    ttft_p95 = percentile(ttfts, 95)
    ttft_p99 = percentile(ttfts, 99)
    itl_p50 = percentile(all_itls, 50)
    itl_p95 = percentile(all_itls, 95)
    itl_p99 = percentile(all_itls, 99)

    ttft_jitter_ratio = (ttft_p99 / ttft_p50) if ttft_p50 > 0.0 else 1.0
    itl_jitter_ratio = (itl_p99 / itl_p50) if itl_p50 > 0.0 else 1.0
    completed_ok = sum(
        1 for s in samples if s.finish_reason in ("stop", "length", "tool_calls") and s.completion_tokens > 0
    )
    completion_rate = (completed_ok / len(samples)) if samples else 1.0

    itl_stability = 1.0 / (1.0 + max(0.0, itl_jitter_ratio - 1.0) * 0.25)
    ttft_stability = 1.0 / (1.0 + max(0.0, ttft_jitter_ratio - 1.0) * 0.20)
    balance_score = round(
        max(0.0, min(1.0, 0.40 * completion_rate + 0.35 * itl_stability + 0.25 * ttft_stability)),
        4,
    )
    if balance_score >= 0.90:
        stability_tier = "optimal"
    elif balance_score >= 0.75:
        stability_tier = "balanced"
    else:
        stability_tier = "degraded"

    return {
        "configuration": configuration,
        "context_tier": context_tier,
        "concurrency": concurrency,
        "requests_completed": len(samples),
        "wall_time_secs": round(wall_time_secs, 4),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "output_tokens_per_sec": round(total_completion / safe_wall, 2),
        "total_tokens_per_sec": round((total_prompt + total_completion) / safe_wall, 2),
        "ttft_ms": {
            "p50": round(ttft_p50, 3),
            "p95": round(ttft_p95, 3),
            "p99": round(ttft_p99, 3),
        },
        "tpot_ms": {
            "p50": round(percentile(tpots, 50), 3),
            "p95": round(percentile(tpots, 95), 3),
            "p99": round(percentile(tpots, 99), 3),
        },
        "itl_ms": {
            "p50": round(itl_p50, 3),
            "p95": round(itl_p95, 3),
            "p99": round(itl_p99, 3),
        },
        "e2e_latency_ms": {
            "p50": round(percentile(latencies, 50), 3),
            "p95": round(percentile(latencies, 95), 3),
            "p99": round(percentile(latencies, 99), 3),
        },
        "ttft_jitter_ratio": round(ttft_jitter_ratio, 4),
        "itl_jitter_ratio": round(itl_jitter_ratio, 4),
        "optimal_runtime_experience": {
            "balance_score": balance_score,
            "itl_jitter_ratio": round(itl_jitter_ratio, 4),
            "ttft_jitter_ratio": round(ttft_jitter_ratio, 4),
            "completion_rate": round(completion_rate, 4),
            "stability_tier": stability_tier,
        },
    }


def measure_streaming_request(
    base_url: str,
    prompt_text: str,
    max_tokens: int,
    *,
    model: str = MODEL_ID,
    api_key: Optional[str] = None,
    dspark: Optional[bool] = None,
    timeout: float = 120.0,
) -> RequestSample:
    """Measure TTFT, per-token ITL, and total latency over an SSE streaming chat completion."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload: Dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    if dspark is not None:
        payload["dspark"] = dspark
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    t0 = time.perf_counter()
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    itl_ms_list: List[float] = []
    completion_tokens = 0
    finish_reason = "stop"

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                break
            now = time.perf_counter()
            try:
                chunk = json.loads(data_str)
            except Exception:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
            content_piece = delta.get("content")
            if content_piece:
                completion_tokens += 1
                if first_token_time is None:
                    first_token_time = now
                elif last_token_time is not None:
                    itl_ms_list.append((now - last_token_time) * 1000.0)
                last_token_time = now

    t_end = time.perf_counter()
    total_ms = (t_end - t0) * 1000.0
    ttft_ms = ((first_token_time - t0) * 1000.0) if first_token_time is not None else total_ms
    approx_prompt_tokens = max(1, len(prompt_text.split()))
    return RequestSample(
        ttft_ms=ttft_ms,
        total_latency_ms=total_ms,
        prompt_tokens=approx_prompt_tokens,
        completion_tokens=max(1, completion_tokens),
        itl_ms_list=itl_ms_list,
        finish_reason=finish_reason,
    )


def run_correctness_gate(
    base_url: str,
    *,
    model: str = MODEL_ID,
    api_key: Optional[str] = None,
) -> Dict[str, object]:
    """Execute the Section 12.1 Functional & Correctness Gate before benchmarking."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Deterministic parity check."}],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": False,
    }
    outputs: List[str] = []
    for _ in range(2):
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            outputs.append(body["choices"][0]["message"]["content"])

    deterministic_match = len(outputs) == 2 and outputs[0] == outputs[1]
    return {
        "passed": deterministic_match,
        "deterministic_greedy_match": deterministic_match,
        "sample_output": outputs[0] if outputs else "",
    }


def _execute_endpoint_matrix(
    endpoint_url: str,
    *,
    model: str,
    api_key: Optional[str],
    configs: List[str],
    selected_tiers: List[str],
    selected_conc: List[int],
    max_tokens: int,
    toggle_dspark: bool,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for cfg_name in configs:
        dspark_flag: Optional[bool] = None
        if toggle_dspark:
            dspark_flag = False if cfg_name == "A_base" else True
        for tier_name in selected_tiers:
            target_words = max(8, min(CONTEXT_TIERS.get(tier_name, 128) // 4, 2048))
            prompt_text = " ".join(["benchmark"] * target_words)
            conc_list = selected_conc if cfg_name == "D_scaling" else [1]
            for conc in conc_list:
                t_start = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                    futures = [
                        pool.submit(
                            measure_streaming_request,
                            endpoint_url,
                            prompt_text,
                            max_tokens,
                            model=model,
                            api_key=api_key,
                            dspark=dspark_flag,
                        )
                        for _ in range(conc)
                    ]
                    samples = [f.result() for f in futures]
                wall = time.perf_counter() - t_start
                summary = summarize_samples(samples, wall, cfg_name, tier_name, conc)
                summary["dspark_enabled"] = dspark_flag
                results.append(summary)
    return results


def run_benchmark_suite(
    base_url: str,
    *,
    deepinfer_url: Optional[str] = None,
    model: str = MODEL_ID,
    api_key: Optional[str] = None,
    configurations: Optional[List[str]] = None,
    tiers: Optional[List[str]] = None,
    concurrencies: Optional[List[int]] = None,
    max_tokens: int = 16,
    output_path: Optional[str | Path] = None,
) -> Dict[str, object]:
    """Run the controlled benchmark suite against a live SINGULARITY endpoint and optional DeepInfer reference."""
    gate = run_correctness_gate(base_url, model=model, api_key=api_key)
    if not gate["passed"]:
        raise RuntimeError(f"Correctness gate failed before benchmark: {gate}")

    configs = configurations or ["A_base", "B_optimized", "C_serving", "D_scaling"]
    selected_tiers = tiers or ["short_128"]
    selected_conc = concurrencies or [1, 4]

    results = _execute_endpoint_matrix(
        base_url,
        model=model,
        api_key=api_key,
        configs=configs,
        selected_tiers=selected_tiers,
        selected_conc=selected_conc,
        max_tokens=max_tokens,
        toggle_dspark=True,
    )

    report: Dict[str, object] = {
        "model_id": model,
        "revision": PINNED_REVISION,
        "hardware": GPU_SPEC,
        "timestamp": time.time(),
        "correctness_gate": gate,
        "benchmarks": results,
    }
    if deepinfer_url:
        deepinfer_results = _execute_endpoint_matrix(
            deepinfer_url,
            model=model,
            api_key=None,
            configs=configs,
            selected_tiers=selected_tiers,
            selected_conc=selected_conc,
            max_tokens=max_tokens,
            toggle_dspark=False,
        )
        report["deepinfer_benchmarks"] = deepinfer_results

    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

