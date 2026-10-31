"""Operator-side tooling: provider-state audit (orphans/zero check), status, cold-start
measurement, and on-demand GPU sampling.

Everything that talks to the provider goes through `ModalProvider`, whose command runner
is injectable. Tests mock the runner; real use shells out to the `modal` CLI. The audit
NEVER reports "verified zero" unless it actually listed provider state with credentials
(`verified_against_provider` is False otherwise).

Provider JSON field names (`Container ID`, `App Name`, ...) follow the `modal ... --json`
CLI output as understood at writing time; they have NOT been verified against a live
account in this workspace. Parsing is tolerant (case/space-insensitive keys).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .config import (
    GPU_COUNT,
    KNOWN_APP_NAMES,
    MODEL_ID,
    ScalingPolicy,
)
from .lifecycle import retry_bounded

Runner = Callable[[Sequence[str]], str]


class ProviderUnavailable(RuntimeError):
    """Credentials/CLI missing, or provider call failed after bounded retries."""


def _norm(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())


def _get(row: Dict[str, object], *names: str) -> Optional[object]:
    table = {_norm(k): v for k, v in row.items()}
    for name in names:
        if _norm(name) in table:
            return table[_norm(name)]
    return None


def _default_runner(args: Sequence[str]) -> str:
    exe = shutil.which("modal")
    if exe is None:
        raise ProviderUnavailable("`modal` CLI not found on PATH")
    proc = subprocess.run([exe, *args], capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"modal {' '.join(args)} exited {proc.returncode}: {proc.stderr.strip()[-300:]}")
    return proc.stdout


class ModalProvider:
    def __init__(
        self,
        runner: Optional[Runner] = None,
        env: Optional[Dict[str, str]] = None,
        home: Optional[Path] = None,
        sleep: Callable[[float], None] = time.sleep,
        assume_credentials: bool = False,
    ) -> None:
        self._runner = runner
        self._assume = assume_credentials  # tests only: skip the credential probe
        self._env = dict(os.environ) if env is None else env
        self._home = home if home is not None else Path.home()
        self._sleep = sleep

    def credentials(self) -> Dict[str, object]:
        if self._assume:
            return {"available": True, "source": "assumed (mock)"}
        if self._env.get("MODAL_TOKEN_ID") and self._env.get("MODAL_TOKEN_SECRET"):
            return {"available": True, "source": "env"}
        cfg = self._home / ".modal.toml"
        if cfg.is_file() and "token_id" in cfg.read_text(errors="ignore"):
            return {"available": True, "source": str(cfg)}
        return {
            "available": False,
            "reason": "no MODAL_TOKEN_ID/MODAL_TOKEN_SECRET and no token in ~/.modal.toml; run `modal setup`",
        }

    def _json(self, args: Sequence[str]) -> List[Dict[str, object]]:
        creds = self.credentials()
        if not creds["available"]:
            raise ProviderUnavailable(str(creds["reason"]))
        runner = self._runner or _default_runner
        try:
            out = retry_bounded(
                lambda: runner(args),
                attempts=3,
                base_delay=1.0,
                max_delay=8.0,
                retry_on=(RuntimeError, subprocess.TimeoutExpired),
                sleep=self._sleep,
            )
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            raise ProviderUnavailable(f"provider call failed after retries: {exc}") from exc
        try:
            data = json.loads(str(out) or "[]")
        except json.JSONDecodeError as exc:
            raise ProviderUnavailable(f"non-JSON provider output: {exc}") from exc
        if not isinstance(data, list):
            raise ProviderUnavailable("unexpected provider JSON shape (expected a list)")
        return [r for r in data if isinstance(r, dict)]

    def list_apps(self) -> List[Dict[str, object]]:
        return self._json(["app", "list", "--json"])

    def list_containers(self) -> List[Dict[str, object]]:
        return self._json(["container", "list", "--json"])

    def stop_container(self, container_id: str) -> str:
        runner = self._runner or _default_runner
        return str(runner(["container", "stop", container_id]))


def _parse_ts(value: object) -> Optional[_dt.datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    for candidate in (text, text.replace(" ", "T", 1)):
        try:
            ts = _dt.datetime.fromisoformat(candidate)
        except ValueError:
            continue
        return ts if ts.tzinfo else ts.replace(tzinfo=_dt.timezone.utc)
    return None


def classify(
    apps: List[Dict[str, object]],
    containers: List[Dict[str, object]],
    policy: ScalingPolicy,
    *,
    now: Optional[_dt.datetime] = None,
    known_apps: Sequence[str] = KNOWN_APP_NAMES,
) -> Dict[str, object]:
    """Pure function: provider rows -> findings. No I/O."""
    now = now or _dt.datetime.now(_dt.timezone.utc)
    app_state: Dict[str, str] = {}
    app_by_id: Dict[str, str] = {}
    for a in apps:
        name = str(_get(a, "Description", "App Name", "name") or "")
        aid = str(_get(a, "App ID", "id") or "")
        state = str(_get(a, "State", "status") or "").lower()
        app_state[name] = state
        if aid:
            app_by_id[aid] = name
    # Absolute ceiling for a legitimate container. The lifetime clock starts at READY, so
    # startup time is added on top of: lifetime cap + idle window + drain/slack.
    ceiling = (
        policy.startup_timeout_secs
        + policy.max_worker_lifetime_secs
        + policy.scaledown_window_secs
        + policy.drain_timeout_secs
        + 300
    )
    billable: List[Dict[str, object]] = []
    orphans: List[Dict[str, object]] = []
    findings: List[str] = []
    for c in containers:
        cid = str(_get(c, "Container ID", "id", "task id") or "?")
        aid = str(_get(c, "App ID") or "")
        name = str(_get(c, "App Name", "app") or app_by_id.get(aid, ""))
        started = _parse_ts(_get(c, "Start Time", "created at", "started"))
        age = (now - started).total_seconds() if started else None
        row = {"container_id": cid, "app": name, "age_secs": age}
        ours = name in known_apps
        row["ours"] = ours
        billable.append(row)
        reasons: List[str] = []
        if ours and app_state.get(name, "").startswith(("stopped", "stopping")):
            reasons.append("app is stopped but container still running")
        if ours and name not in app_state and apps:
            reasons.append("container's app not present in app list")
        if ours and age is not None and age > ceiling:
            reasons.append(f"age {int(age)}s exceeds lifetime ceiling {ceiling}s (no idle shutdown happened)")
        if ours and age is None:
            reasons.append("start time unparseable; cannot bound age")
        if reasons:
            orphans.append({**row, "reasons": reasons})
    if policy.pinned_warm:
        findings.append("policy.pinned_warm is True: provider keeps min_containers billing 8x B200 forever")
    ours_running = [r for r in billable if r["ours"]]
    return {
        "billable_containers": billable,
        "our_billable_containers": len(ours_running),
        "foreign_containers": len(billable) - len(ours_running),
        "orphans": orphans,
        "findings": findings,
        "zero_inference_compute": len(ours_running) == 0,
        "lifetime_ceiling_secs": ceiling,
    }


def audit(
    provider: ModalProvider,
    policy: ScalingPolicy,
    *,
    terminate_orphans: bool = False,
    now: Optional[_dt.datetime] = None,
) -> Dict[str, object]:
    """Provider-side zero check + orphan detection. Dry-run unless `terminate_orphans`."""
    creds = provider.credentials()
    base: Dict[str, object] = {
        "verified_against_provider": False,
        "credentials": creds,
        "policy": {
            "min_containers": policy.min_containers,
            "max_containers": policy.max_containers,
            "scaledown_window_secs": policy.scaledown_window_secs,
            "max_worker_lifetime_secs": policy.max_worker_lifetime_secs,
            "max_spend_usd": policy.max_spend_usd,
            "source": "local_env (use `status --url <endpoint>` to inspect live container limits)",
        },
    }
    if not creds["available"]:
        base.update(
            status="unverifiable",
            note="Modal credentials unavailable: provider state was NOT checked; "
            "zero-compute is UNKNOWN, not confirmed.",
        )
        return base
    try:
        apps = provider.list_apps()
        containers = provider.list_containers()
    except ProviderUnavailable as exc:
        base.update(status="unverifiable", note=f"provider query failed: {exc}")
        return base
    result = classify(apps, containers, policy, now=now)
    base.update(result)
    base["verified_against_provider"] = True
    terminated: List[Dict[str, object]] = []
    if terminate_orphans:
        for o in result["orphans"]:  # type: ignore[union-attr]
            try:
                provider.stop_container(str(o["container_id"]))
                terminated.append({"container_id": o["container_id"], "ok": True})
            except Exception as exc:  # noqa: BLE001
                terminated.append({"container_id": o["container_id"], "ok": False, "error": str(exc)})
    base["terminated"] = terminated
    base["status"] = "clean" if not result["orphans"] and not result["findings"] else "attention"
    return base


# ---- on-demand GPU sampling (never polled in the background) ---------------------

_PEAK: Dict[str, float] = {"vram_mib": 0.0}


def gpu_snapshot(runner: Optional[Callable[[], str]] = None) -> Dict[str, object]:
    """One nvidia-smi sample: per-GPU utilization and VRAM; updates process-wide peak."""

    def _smi() -> str:
        return subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout

    try:
        text = (runner or _smi)()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "util_pct": float(parts[1]),
                    "vram_used_mib": float(parts[2]),
                    "vram_total_mib": float(parts[3]),
                }
            )
        except ValueError:
            continue
    total_used = sum(g["vram_used_mib"] for g in gpus)
    _PEAK["vram_mib"] = max(_PEAK["vram_mib"], max((g["vram_used_mib"] for g in gpus), default=0.0))
    return {
        "available": bool(gpus),
        "gpus": gpus,
        "mean_util_pct": statistics.fmean(g["util_pct"] for g in gpus) if gpus else None,
        "total_vram_used_mib": total_used,
        "peak_single_gpu_vram_mib_observed": _PEAK["vram_mib"],
        "measured": True,
    }


# ---- cold-start measurement ------------------------------------------------------


def measure_cold_start(
    base_url: str,
    *,
    model: str = MODEL_ID,
    api_key: Optional[str] = None,
    warm_requests: int = 3,
    max_tokens: int = 16,
    timeout: float = 3600.0,
    gpu_count: int = GPU_COUNT,
    sampler: Optional[Callable[..., object]] = None,
) -> Dict[str, object]:
    """Time the FIRST request (which pays provisioning + model init if the app was at
    zero) and N follow-ups. All values are wall-clock measurements from THIS client.

    Whether the first request really hit zero is operator-asserted: run `deploy.py audit`
    first and confirm `zero_inference_compute` is true. Cold overhead is derived
    (first TTFT minus warm median), so it includes network RTT noise.
    """
    if not 0 <= warm_requests <= 50:
        raise ValueError("warm_requests must be in [0, 50]")
    if sampler is None:
        from .benchmark import measure_streaming_request as sampler  # lazy: avoids import cycle cost

    def _one(prompt: str) -> Dict[str, float]:
        s = sampler(base_url, prompt, max_tokens, model=model, api_key=api_key, timeout=timeout)  # type: ignore[misc]
        return {
            "ttft_ms": s.ttft_ms,
            "total_ms": s.total_latency_ms,
            "tpot_ms": s.tpot_ms,
            "completion_tokens": s.completion_tokens,
        }

    t0 = time.perf_counter()
    first = _one("Say hello.")
    first["wall_ms"] = (time.perf_counter() - t0) * 1000.0
    warm = [_one("Say hello again.") for _ in range(warm_requests)]
    warm_ttft = [w["ttft_ms"] for w in warm]
    report: Dict[str, object] = {
        "kind": "measured",
        "base_url": base_url,
        "first_request": first,
        "warm_requests": warm,
        "warm_ttft_median_ms": statistics.median(warm_ttft) if warm_ttft else None,
        "cold_overhead_ms": (first["ttft_ms"] - statistics.median(warm_ttft)) if warm_ttft else None,
        "gpu_count": gpu_count,
        "caveat": "first request is only a cold start if the app was at zero beforehand "
        "(verify with `deploy.py audit`); provisioning vs model-init split is not separable "
        "from the client side -- see startup_log timestamps in container logs.",
    }
    return report


def status(provider: ModalProvider, policy: ScalingPolicy, *, url: Optional[str] = None) -> Dict[str, object]:
    """Operator status: policy, provider zero-check, and (optionally) in-container lifecycle."""
    out: Dict[str, object] = {"audit": audit(provider, policy)}
    if url:
        from .health import _http_json  # local import keeps module import light

        try:
            code, body = _http_json(f"{url.rstrip('/')}/__lifecycle/status", timeout=30.0)
            out["container"] = {"http": code, "lifecycle": body}
            out["warning"] = "querying a live endpoint counts as activity and resets the idle window"
        except Exception as exc:  # noqa: BLE001
            out["container"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out
