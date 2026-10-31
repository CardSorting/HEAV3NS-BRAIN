"""Container-side lifecycle control for the scale-to-zero SINGULARITY service.

Pure stdlib, no Modal/GPU dependency, so every rule here is unit-testable offline.

Scope: this governs ONE container. "Zero" (no container at all) is not a state this code
can observe -- it is the absence of the process. The provider releases the container
after `scaledown_window` idle seconds; this module guarantees the container never gets
in the way of that (no phantom in-flight work, bounded drain, bounded lifetime/spend).

States and legal transitions::

    STARTING -> INITIALIZING -> READY <-> (READY)
    READY -> DRAINING -> STOPPED
    STARTING/INITIALIZING/READY/DRAINING -> FAILED -> STOPPED
    STOPPED is terminal.
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Type


class LifecycleState(str, enum.Enum):
    STARTING = "starting"
    INITIALIZING = "initializing"
    READY = "ready"
    DRAINING = "draining"
    FAILED = "failed"
    STOPPED = "stopped"


_TRANSITIONS: Dict[LifecycleState, Tuple[LifecycleState, ...]] = {
    LifecycleState.STARTING: (
        LifecycleState.INITIALIZING,
        LifecycleState.FAILED,
        LifecycleState.DRAINING,
    ),
    LifecycleState.INITIALIZING: (
        LifecycleState.READY,
        LifecycleState.FAILED,
        LifecycleState.DRAINING,
    ),
    LifecycleState.READY: (LifecycleState.DRAINING, LifecycleState.FAILED),
    LifecycleState.DRAINING: (LifecycleState.STOPPED, LifecycleState.FAILED),
    LifecycleState.FAILED: (LifecycleState.STOPPED,),
    LifecycleState.STOPPED: (),
}


class IllegalTransition(RuntimeError):
    pass


class AdmissionRejected(Exception):
    """Raised when a request must not be forwarded. `code` maps to an HTTP status."""

    STATUS = {
        "not_ready": 503,
        "draining": 503,
        "concurrency": 429,
        "lifetime": 503,
        "spend": 503,
        "duplicate": 409,
    }

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail

    @property
    def http_status(self) -> int:
        return self.STATUS.get(self.code, 503)


@dataclass
class _Job:
    request_id: str
    started: float
    deadline: float


@dataclass
class LifecycleTracker:
    """Thread-safe state machine + in-flight accounting.

    `max_concurrent`, `request_timeout_secs`, `max_lifetime_secs`, and the optional spend
    guard are all enforced at admission time. In-flight work is counted from admission
    until `release()` -- including streaming responses, which must release in a
    `finally` after the stream closes -- so an empty queue is never mistaken for idle.
    """

    max_concurrent: int = 16
    request_timeout_secs: float = 1800.0
    max_lifetime_secs: float = 6 * 3600.0
    gpu_count: int = 8
    gpu_usd_per_gpu_hour: Optional[float] = None
    max_spend_usd: Optional[float] = None
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self._cond = threading.Condition()
        self._state = LifecycleState.STARTING
        self._born = self.clock()
        self._last_activity = self._born
        self._ready_at: Optional[float] = None
        self._jobs: Dict[str, _Job] = {}
        self._history: list[Tuple[float, str]] = [(0.0, self._state.value)]
        self.drain_reason: Optional[str] = None
        self.completed = 0
        self.failed = 0
        self.timed_out = 0
        self.rejected: Dict[str, int] = {}
        self.busy_seconds = 0.0  # sum of per-request wall time (request-level accounting)
        self.tokens_out = 0

    # ---- state machine -------------------------------------------------
    @property
    def state(self) -> LifecycleState:
        with self._cond:
            return self._state

    def transition(self, to: LifecycleState) -> LifecycleState:
        """Idempotent for same-state; raises IllegalTransition otherwise-invalid moves."""
        with self._cond:
            if to == self._state:
                return self._state
            if to not in _TRANSITIONS[self._state]:
                raise IllegalTransition(f"{self._state.value} -> {to.value} is not allowed")
            self._state = to
            if to == LifecycleState.READY and self._ready_at is None:
                self._ready_at = self.clock()
            self._history.append((self.clock() - self._born, to.value))
            self._cond.notify_all()
            return to

    # ---- cost accounting -----------------------------------------------
    def uptime(self) -> float:
        return self.clock() - self._born

    def estimated_spend_usd(self) -> Optional[float]:
        """Container-lifetime GPU cost ESTIMATE (None when no price was supplied)."""
        if self.gpu_usd_per_gpu_hour is None:
            return None
        return self.uptime() / 3600.0 * self.gpu_count * self.gpu_usd_per_gpu_hour

    # ---- admission / in-flight ----------------------------------------
    def _reject(self, code: str, detail: str) -> AdmissionRejected:
        self.rejected[code] = self.rejected.get(code, 0) + 1
        return AdmissionRejected(code, detail)

    def serving_uptime(self) -> float:
        """Seconds since the worker became READY (0.0 before that)."""
        with self._cond:
            if self._ready_at is None:
                return 0.0
            return self.clock() - self._ready_at

    def limit_reason(self) -> Optional[str]:
        """'lifetime' / 'spend' if a hard guard has tripped, else None.

        `max_lifetime_secs` is a *serving* lifetime: the clock starts at READY so a slow
        model load can never eat the serving budget (a 120 s guard would otherwise trip the
        instant a 10-minute init finished). Spend is deliberately still measured from
        container start, because billing starts then. Pre-READY time is bounded separately by
        the provider's `startup_timeout`.
        """
        if self.serving_uptime() >= self.max_lifetime_secs:
            return "lifetime"
        spend = self.estimated_spend_usd()
        if self.max_spend_usd is not None and spend is not None and spend >= self.max_spend_usd:
            return "spend"
        return None

    def admit(self, request_id: str) -> None:
        with self._cond:
            if self._state in (LifecycleState.DRAINING, LifecycleState.STOPPED, LifecycleState.FAILED):
                raise self._reject("draining", f"worker is {self._state.value}; retry elsewhere")
            if self._state != LifecycleState.READY:
                raise self._reject("not_ready", f"worker is {self._state.value}")
            reason = self.limit_reason()
            if reason == "lifetime":
                self._begin_drain_locked("lifetime")
                raise self._reject("lifetime", "max worker lifetime reached; draining")
            if reason == "spend":
                self._begin_drain_locked("spend")
                raise self._reject("spend", f"spend limit ${self.max_spend_usd:.2f} reached; draining")
            if request_id in self._jobs:
                raise self._reject("duplicate", f"request {request_id} already in flight")
            if len(self._jobs) >= self.max_concurrent:
                raise self._reject("concurrency", f"max concurrent requests {self.max_concurrent}")
            now = self.clock()
            self._jobs[request_id] = _Job(request_id, now, now + self.request_timeout_secs)
            self._last_activity = now

    def release(self, request_id: str, *, ok: bool = True, tokens: int = 0) -> bool:
        """Idempotent: releasing an unknown/already released id returns False."""
        with self._cond:
            job = self._jobs.pop(request_id, None)
            if job is None:
                return False
            now = self.clock()
            self.busy_seconds += now - job.started
            self.tokens_out += max(0, tokens)
            if ok:
                self.completed += 1
            else:
                self.failed += 1
            self._last_activity = now
            self._cond.notify_all()
            return True

    def expired_jobs(self) -> list[str]:
        """Ids past their per-job deadline (stuck jobs); caller cancels + releases them."""
        with self._cond:
            now = self.clock()
            return [j.request_id for j in self._jobs.values() if now >= j.deadline]

    def reap_expired(self) -> list[str]:
        ids = self.expired_jobs()
        for rid in ids:
            if self.release(rid, ok=False):
                with self._cond:
                    self.timed_out += 1
        return ids

    @property
    def in_flight(self) -> int:
        with self._cond:
            return len(self._jobs)

    def idle_seconds(self) -> float:
        with self._cond:
            return 0.0 if self._jobs else self.clock() - self._last_activity

    def is_idle(self) -> bool:
        with self._cond:
            return not self._jobs

    # ---- drain ---------------------------------------------------------
    def _begin_drain_locked(self, reason: Optional[str] = None) -> None:
        if self._state in (LifecycleState.DRAINING, LifecycleState.STOPPED, LifecycleState.FAILED):
            return
        self._state = LifecycleState.DRAINING
        self.drain_reason = reason
        self._history.append((self.clock() - self._born, self._state.value))
        self._cond.notify_all()

    def begin_drain(self, reason: Optional[str] = None) -> None:
        """Stop admitting new work. Safe to call repeatedly / concurrently."""
        with self._cond:
            self._begin_drain_locked(reason)

    def drain(self, timeout_secs: float) -> bool:
        """Block until in-flight == 0 or timeout. True iff fully drained.

        Always bounded: after the timeout the caller proceeds to terminate the child
        process anyway (deterministic shutdown beats waiting forever), and the
        abandoned requests are counted as failed.
        """
        self.begin_drain()
        deadline = self.clock() + timeout_secs
        with self._cond:
            while self._jobs:
                remaining = deadline - self.clock()
                if remaining <= 0:
                    abandoned = list(self._jobs)
                    for rid in abandoned:
                        self._jobs.pop(rid, None)
                        self.failed += 1
                    return False
                self._cond.wait(timeout=min(remaining, 0.25))
            return True

    # ---- observability -------------------------------------------------
    def snapshot(self) -> Dict[str, object]:
        with self._cond:
            now = self.clock()
            return {
                "state": self._state.value,
                "uptime_secs": round(now - self._born, 3),
                "serving_uptime_secs": (
                    0.0 if self._ready_at is None else round(now - self._ready_at, 3)
                ),
                "in_flight": len(self._jobs),
                "idle_secs": 0.0 if self._jobs else round(now - self._last_activity, 3),
                "max_concurrent": self.max_concurrent,
                # Effective in-container limits (what is actually enforced, not the local env).
                "limits": {
                    "max_lifetime_secs": self.max_lifetime_secs,
                    "max_spend_usd": self.max_spend_usd,
                    "request_timeout_secs": self.request_timeout_secs,
                },
                "completed": self.completed,
                "failed": self.failed,
                "timed_out": self.timed_out,
                "rejected": dict(self.rejected),
                "busy_seconds": round(self.busy_seconds, 3),
                "tokens_out": self.tokens_out,
                "gpu_seconds_provisioned": round((now - self._born) * self.gpu_count, 3),
                "gpu_seconds_busy_upper_bound": round(self.busy_seconds * self.gpu_count, 3),
                "estimated_spend_usd": (
                    None
                    if self.gpu_usd_per_gpu_hour is None
                    else round((now - self._born) / 3600.0 * self.gpu_count * self.gpu_usd_per_gpu_hour, 4)
                ),
                "spend_is_estimate": True,
                "history": list(self._history),
            }


def retry_bounded(
    fn: Callable[[], object],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: Tuple[Type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
) -> object:
    """Call `fn` at most `attempts` times with capped exponential backoff, then raise.

    There is deliberately no unbounded mode.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last: Optional[BaseException] = None
    for i in range(attempts):
        try:
            return fn()
        except retry_on as exc:  # noqa: PERF203
            last = exc
            if i + 1 < attempts:
                sleep(min(max_delay, base_delay * (2**i)))
    assert last is not None
    raise last


@dataclass
class StartupResult:
    ok: bool
    attempts: int = 0
    error: Optional[str] = None
    cleaned_up: bool = False
    extra: Dict[str, object] = field(default_factory=dict)


def start_with_cleanup(
    tracker: LifecycleTracker,
    start: Callable[[], object],
    stop: Callable[[], None],
    *,
    attempts: int = 1,
    sleep: Callable[[float], None] = time.sleep,
) -> StartupResult:
    """Run startup under the state machine; on any failure tear down partial init.

    `stop` must be idempotent. The caller decides whether to re-raise (the Modal
    `@enter` must re-raise so the container fails instead of serving a half-initialised
    cluster).
    """
    tracker.transition(LifecycleState.INITIALIZING)
    tries = 0
    last_err: Optional[str] = None
    for i in range(max(1, attempts)):
        tries += 1
        try:
            start()
            tracker.transition(LifecycleState.READY)
            return StartupResult(True, tries)
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            try:
                stop()
            finally:
                pass
            if i + 1 < attempts:
                sleep(min(30.0, 2.0**i))
    tracker.transition(LifecycleState.FAILED)
    return StartupResult(False, tries, last_err, cleaned_up=True)


GUARD_DRAIN_REASONS = ("lifetime", "spend")


class LifecycleWatchdog:
    """Turns the lifetime/spend guards (and a dead child) into an actual container exit.

    Admission control alone only refuses work; an idle container that tripped a guard
    would otherwise keep billing until the provider's idle window expired (and a
    pinned-warm one forever). The watchdog polls LOCAL state only -- tracker counters and
    child-process liveness; it never touches the GPUs or the upstream server -- so it
    cannot itself keep compute busy. Exit path: stop admitting -> wait (bounded by
    `drain_timeout_secs`) for in-flight work -> abandon the rest (counted as failed) ->
    `stop_child()` -> STOPPED -> `exit_fn()` (the Modal container sends itself SIGINT,
    which runs the `@modal.exit` hook and releases the container).

    `check()` is pure and synchronous so it is unit-testable with a fake clock.
    """

    def __init__(
        self,
        tracker: LifecycleTracker,
        *,
        drain_timeout_secs: float,
        stop_child: Callable[[], None],
        exit_fn: Callable[[], None],
        liveness: Optional[Callable[[], bool]] = None,
        poll_secs: float = 5.0,
    ) -> None:
        self.tracker = tracker
        self.drain_timeout_secs = drain_timeout_secs
        self.stop_child = stop_child
        self.exit_fn = exit_fn
        self.liveness = liveness
        self.poll_secs = poll_secs
        self.stop_reason: Optional[str] = None
        self._drain_deadline: Optional[float] = None
        self._lock = threading.Lock()
        self._halt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def check(self) -> Optional[str]:
        """One evaluation. Returns the stop reason once the container has been stopped."""
        with self._lock:
            if self.stop_reason is not None:
                return self.stop_reason
            t = self.tracker
            t.reap_expired()
            if t.state == LifecycleState.READY:
                if self.liveness is not None and not self.liveness():
                    t.transition(LifecycleState.FAILED)
                else:
                    reason = t.limit_reason()
                    if reason is not None:
                        t.begin_drain(reason)
            state = t.state
            if state == LifecycleState.FAILED:
                return self._stop("worker_failed")
            if state == LifecycleState.DRAINING and t.drain_reason in GUARD_DRAIN_REASONS:
                now = t.clock()
                if self._drain_deadline is None:
                    self._drain_deadline = now + self.drain_timeout_secs
                if t.in_flight == 0 or now >= self._drain_deadline:
                    t.drain(0)  # abandons (and counts) anything still in flight
                    return self._stop(str(t.drain_reason))
            return None

    def _stop(self, reason: str) -> str:
        self.stop_reason = reason
        try:
            self.stop_child()
        finally:
            try:
                self.tracker.transition(LifecycleState.STOPPED)
            except IllegalTransition:
                pass
            self.exit_fn()
        return reason

    def start(self) -> None:
        if self._thread is not None:
            return

        def _run() -> None:
            while not self._halt.wait(self.poll_secs):
                try:
                    if self.check() is not None:
                        return
                except Exception as exc:  # noqa: BLE001 - never let the watchdog die silently
                    print(f"[SINGULARITY watchdog] check failed: {type(exc).__name__}: {exc}", flush=True)

        self._thread = threading.Thread(target=_run, name="lifecycle-watchdog", daemon=True)
        self._thread.start()

    def halt(self) -> None:
        self._halt.set()
