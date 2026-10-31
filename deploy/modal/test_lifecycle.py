"""Offline tests for lifecycle, audit/orphan detection, and measurement tooling.

No Modal credentials, GPU, or network are used. Provider calls are mocked through the
injectable runner of `ModalProvider`; nothing here proves behavior against real Modal.
"""

import datetime as dt
import json
import sys
import threading
import unittest
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
_MODAL_DIR = str(Path(__file__).resolve().parent)
if _REPO_ROOT in sys.path and _MODAL_DIR in sys.path[: sys.path.index(_REPO_ROOT)]:
    sys.path.remove(_REPO_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_loaded = sys.modules.get("deploy")
if _loaded is not None and not hasattr(_loaded, "__path__"):
    del sys.modules["deploy"]

from deploy.modal import ops  # noqa: E402
from deploy.modal.config import resolve_scaling_policy  # noqa: E402
from deploy.modal.deploy import SCALE_TO_ZERO_COMMANDS, execute_command  # noqa: E402
from deploy.modal.lifecycle import (  # noqa: E402
    AdmissionRejected,
    IllegalTransition,
    LifecycleState,
    LifecycleTracker,
    LifecycleWatchdog,
    retry_bounded,
    start_with_cleanup,
)
import time as _time  # noqa: E402


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, secs: float) -> None:
        self.t += secs


def ready_tracker(clock=None, **kw) -> LifecycleTracker:
    t = LifecycleTracker(clock=clock or FakeClock(), **kw)
    t.transition(LifecycleState.INITIALIZING)
    t.transition(LifecycleState.READY)
    return t


class StateMachineTests(unittest.TestCase):
    def test_illegal_transitions_rejected(self):
        t = LifecycleTracker(clock=FakeClock())
        with self.assertRaises(IllegalTransition):
            t.transition(LifecycleState.READY)  # must pass through INITIALIZING
        t.transition(LifecycleState.FAILED)
        t.transition(LifecycleState.STOPPED)
        with self.assertRaises(IllegalTransition):
            t.transition(LifecycleState.READY)

    def test_same_state_transition_is_idempotent(self):
        t = ready_tracker()
        self.assertEqual(t.transition(LifecycleState.READY), LifecycleState.READY)

    def test_not_ready_rejects_work(self):
        t = LifecycleTracker(clock=FakeClock())
        with self.assertRaises(AdmissionRejected) as cm:
            t.admit("a")
        self.assertEqual(cm.exception.code, "not_ready")


class AdmissionTests(unittest.TestCase):
    def test_concurrency_cap_and_duplicate(self):
        t = ready_tracker(max_concurrent=2)
        t.admit("a")
        t.admit("b")
        with self.assertRaises(AdmissionRejected) as cm:
            t.admit("c")
        self.assertEqual((cm.exception.code, cm.exception.http_status), ("concurrency", 429))
        with self.assertRaises(AdmissionRejected) as cm:
            t.admit("a")
        self.assertEqual(cm.exception.code, "duplicate")
        self.assertTrue(t.release("a"))
        self.assertFalse(t.release("a"), "release is idempotent")
        t.admit("c")

    def test_empty_queue_is_not_idle_while_in_flight(self):
        clock = FakeClock()
        t = ready_tracker(clock)
        t.admit("a")
        clock.advance(500)
        self.assertFalse(t.is_idle())
        self.assertEqual(t.idle_seconds(), 0.0)
        t.release("a", tokens=10)
        clock.advance(30)
        self.assertTrue(t.is_idle())
        self.assertEqual(t.idle_seconds(), 30.0)
        snap = t.snapshot()
        self.assertEqual((snap["completed"], snap["tokens_out"]), (1, 10))
        self.assertEqual(snap["busy_seconds"], 500.0)

    def test_stuck_job_is_reaped_at_deadline(self):
        clock = FakeClock()
        t = ready_tracker(clock, request_timeout_secs=100)
        t.admit("stuck")
        clock.advance(99)
        self.assertEqual(t.reap_expired(), [])
        clock.advance(2)
        self.assertEqual(t.reap_expired(), ["stuck"])
        self.assertEqual((t.in_flight, t.timed_out, t.failed), (0, 1, 1))

    def test_max_lifetime_triggers_drain(self):
        clock = FakeClock()
        t = ready_tracker(clock, max_lifetime_secs=60)
        clock.advance(61)
        with self.assertRaises(AdmissionRejected) as cm:
            t.admit("a")
        self.assertEqual(cm.exception.code, "lifetime")
        self.assertEqual(t.state, LifecycleState.DRAINING)

    def test_spend_limit(self):
        clock = FakeClock()
        # 8 GPUs * $1/h => $8/h; limit $1 reached after 450s.
        t = ready_tracker(clock, gpu_usd_per_gpu_hour=1.0, max_spend_usd=1.0)
        clock.advance(449)
        t.admit("a")
        t.release("a")
        clock.advance(2)
        with self.assertRaises(AdmissionRejected) as cm:
            t.admit("b")
        self.assertEqual(cm.exception.code, "spend")
        self.assertTrue(t.snapshot()["spend_is_estimate"])

    def test_no_price_means_no_spend_figure(self):
        t = ready_tracker()
        self.assertIsNone(t.estimated_spend_usd())
        self.assertIsNone(t.snapshot()["estimated_spend_usd"])

    def test_concurrent_admission_never_exceeds_cap(self):
        t = ready_tracker(max_concurrent=5)
        admitted, lock = [], threading.Lock()

        def worker(i):
            try:
                t.admit(f"r{i}")
                with lock:
                    admitted.append(i)
            except AdmissionRejected:
                pass

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        [th.start() for th in threads]
        [th.join() for th in threads]
        self.assertEqual(len(admitted), 5)
        self.assertEqual(t.in_flight, 5)


class DrainTests(unittest.TestCase):
    def test_drain_waits_for_in_flight_then_succeeds(self):
        t = ready_tracker(clock=__import__("time").monotonic)
        t.admit("a")
        threading.Timer(0.1, lambda: t.release("a")).start()
        self.assertTrue(t.drain(5.0))
        self.assertEqual(t.state, LifecycleState.DRAINING)
        with self.assertRaises(AdmissionRejected) as cm:
            t.admit("late")
        self.assertEqual(cm.exception.code, "draining")

    def test_drain_is_bounded_and_counts_abandoned(self):
        t = ready_tracker(clock=__import__("time").monotonic)
        t.admit("stuck")
        self.assertFalse(t.drain(0.3))
        self.assertEqual((t.in_flight, t.failed), (0, 1))

    def test_drain_idempotent_and_stoppable(self):
        t = ready_tracker()
        self.assertTrue(t.drain(1))
        self.assertTrue(t.drain(1))
        t.transition(LifecycleState.STOPPED)


class RetryAndStartupTests(unittest.TestCase):
    def test_retry_bounded_gives_up(self):
        calls, sleeps = [], []

        def boom():
            calls.append(1)
            raise RuntimeError("x")

        with self.assertRaises(RuntimeError):
            retry_bounded(boom, attempts=3, base_delay=1, max_delay=3, sleep=sleeps.append)
        self.assertEqual((len(calls), sleeps), (3, [1, 2]))

    def test_retry_bounded_succeeds_midway(self):
        n = {"i": 0}

        def flaky():
            n["i"] += 1
            if n["i"] < 2:
                raise OSError("t")
            return "ok"

        self.assertEqual(retry_bounded(flaky, attempts=3, sleep=lambda s: None), "ok")

    def test_retry_requires_at_least_one_attempt(self):
        with self.assertRaises(ValueError):
            retry_bounded(lambda: 1, attempts=0)

    def test_partial_init_failure_cleans_up_and_fails(self):
        t = LifecycleTracker(clock=FakeClock())
        stopped = []

        def start():
            raise RuntimeError("rank 3 NCCL init failed")

        res = start_with_cleanup(t, start, lambda: stopped.append(1))
        self.assertFalse(res.ok)
        self.assertTrue(res.cleaned_up)
        self.assertEqual(stopped, [1])
        self.assertEqual(t.state, LifecycleState.FAILED)
        with self.assertRaises(AdmissionRejected):
            t.admit("a")

    def test_startup_success_reaches_ready(self):
        t = LifecycleTracker(clock=FakeClock())
        res = start_with_cleanup(t, lambda: None, lambda: None)
        self.assertTrue(res.ok)
        self.assertEqual(t.state, LifecycleState.READY)

    def test_startup_retries_are_bounded(self):
        t = LifecycleTracker(clock=FakeClock())
        n = []
        res = start_with_cleanup(
            t, lambda: (n.append(1), 1 / 0), lambda: None, attempts=2, sleep=lambda s: None
        )
        self.assertEqual((res.ok, res.attempts, len(n)), (False, 2, 2))


class PolicyTests(unittest.TestCase):
    def test_new_limits_defaults_and_bounds(self):
        p = resolve_scaling_policy({})
        self.assertEqual(p.min_containers, 0)
        self.assertIsNone(p.max_spend_usd)
        for env in (
            {"SINGULARITY_MAX_CONCURRENT_REQUESTS": "0"},
            {"SINGULARITY_MAX_CONCURRENT_REQUESTS": "9999"},
            {"SINGULARITY_DRAIN_TIMEOUT_SECS": "0"},
            {"SINGULARITY_MAX_WORKER_LIFETIME_SECS": "10"},
            {"SINGULARITY_MAX_WORKER_LIFETIME_SECS": "999999"},
            {"SINGULARITY_MAX_SPEND_USD": "5"},  # needs a price
            {"SINGULARITY_MAX_SPEND_USD": "-1", "SINGULARITY_GPU_USD_PER_GPU_HOUR": "1"},
            {"SINGULARITY_MAX_SPEND_USD": "nan", "SINGULARITY_GPU_USD_PER_GPU_HOUR": "1"},
        ):
            with self.subTest(env=env), self.assertRaises(ValueError):
                resolve_scaling_policy(env)

    def test_spend_with_price_ok(self):
        p = resolve_scaling_policy(
            {"SINGULARITY_MAX_SPEND_USD": "50", "SINGULARITY_GPU_USD_PER_GPU_HOUR": "3.5"}
        )
        self.assertEqual((p.max_spend_usd, p.gpu_usd_per_gpu_hour), (50.0, 3.5))


NOW = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)


def _iso(delta_secs):
    return (NOW - dt.timedelta(seconds=delta_secs)).isoformat()


def mock_provider(apps, containers, calls=None):
    def runner(args):
        if calls is not None:
            calls.append(list(args))
        if args[:2] == ["app", "list"]:
            return json.dumps(apps)
        if args[:2] == ["container", "list"]:
            return json.dumps(containers)
        if args[:2] == ["container", "stop"]:
            return ""
        raise AssertionError(f"unexpected {args}")

    return ops.ModalProvider(runner=runner, assume_credentials=True, sleep=lambda s: None)


APP = "singularity-v41-flash-1017"


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.policy = resolve_scaling_policy({})

    def test_no_credentials_is_unverifiable_not_zero(self):
        prov = ops.ModalProvider(env={}, home=Path("/nonexistent-home-xyz"))
        rep = ops.audit(prov, self.policy)
        self.assertEqual(rep["status"], "unverifiable")
        self.assertFalse(rep["verified_against_provider"])
        self.assertNotIn("zero_inference_compute", rep)

    def test_zero_containers_verified_zero(self):
        rep = ops.audit(mock_provider([{"App ID": "ap-1", "Description": APP, "State": "deployed"}], []), self.policy, now=NOW)
        self.assertTrue(rep["verified_against_provider"])
        self.assertTrue(rep["zero_inference_compute"])
        self.assertEqual(rep["status"], "clean")

    def test_healthy_running_container_is_billable_not_orphan(self):
        apps = [{"App ID": "ap-1", "Description": APP, "State": "deployed"}]
        cs = [{"Container ID": "ta-1", "App ID": "ap-1", "App Name": APP, "Start Time": _iso(120)}]
        rep = ops.audit(mock_provider(apps, cs), self.policy, now=NOW)
        self.assertFalse(rep["zero_inference_compute"])
        self.assertEqual(rep["orphans"], [])

    def test_orphans_detected_and_terminated_only_on_request(self):
        apps = [{"App ID": "ap-1", "Description": APP, "State": "stopped"}]
        cs = [
            {"Container ID": "ta-dead-app", "App ID": "ap-1", "App Name": APP, "Start Time": _iso(60)},
            {"Container ID": "ta-ancient", "App ID": "ap-2", "App Name": "deepinfer-v41-reference", "Start Time": _iso(10 * 86400)},
            {"Container ID": "ta-foreign", "App ID": "ap-9", "App Name": "someone-else", "Start Time": _iso(10 * 86400)},
        ]
        calls = []
        rep = ops.audit(mock_provider(apps, cs, calls), self.policy, now=NOW)
        ids = {o["container_id"] for o in rep["orphans"]}
        self.assertEqual(ids, {"ta-dead-app", "ta-ancient"})
        self.assertEqual(rep["foreign_containers"], 1)
        self.assertEqual(rep["terminated"], [])
        self.assertFalse(any(c[:2] == ["container", "stop"] for c in calls))

        calls.clear()
        rep = ops.audit(mock_provider(apps, cs, calls), self.policy, terminate_orphans=True, now=NOW)
        stops = [c for c in calls if c[:2] == ["container", "stop"]]
        self.assertEqual({c[2] for c in stops}, {"ta-dead-app", "ta-ancient"})
        self.assertNotIn(["container", "stop", "ta-foreign"], calls)
        self.assertTrue(all(t["ok"] for t in rep["terminated"]))

    def test_pinned_warm_is_flagged(self):
        pol = resolve_scaling_policy({"SINGULARITY_MIN_CONTAINERS": "1", "SINGULARITY_ALLOW_PINNED_WARM": "1"})
        rep = ops.audit(mock_provider([], []), pol, now=NOW)
        self.assertEqual(rep["status"], "attention")
        self.assertTrue(rep["findings"])

    def test_provider_failure_is_bounded_and_unverifiable(self):
        calls = []

        def runner(args):
            calls.append(1)
            raise RuntimeError("503 from api")

        prov = ops.ModalProvider(runner=runner, assume_credentials=True, sleep=lambda s: None)
        rep = ops.audit(prov, self.policy)
        self.assertEqual(rep["status"], "unverifiable")
        self.assertEqual(len(calls), 3)  # bounded retries, then give up

    def test_garbage_provider_output(self):
        prov = ops.ModalProvider(runner=lambda a: "not json", assume_credentials=True, sleep=lambda s: None)
        self.assertEqual(ops.audit(prov, self.policy)["status"], "unverifiable")

    def test_cli_audit_and_status_use_injected_provider(self):
        prov = mock_provider([], [])
        out = execute_command(["audit"], provider=prov)
        self.assertTrue(out["zero_inference_compute"])
        out = execute_command(["status"], provider=prov)
        self.assertTrue(out["audit"]["verified_against_provider"])
        self.assertEqual(execute_command(["--dry-run", "audit", "--terminate-orphans"], provider=prov)["status"], "dry_run")
        self.assertEqual(execute_command(["--dry-run", "teardown"], provider=prov)["status"], "dry_run")
        self.assertEqual(SCALE_TO_ZERO_COMMANDS, ["status", "audit", "measure-cold-start", "teardown"])

    def test_smoke_app_classified_as_ours_and_orphan_caught(self):
        apps = [{"App ID": "ap-s", "Description": "singularity-smoke", "State": "stopped"}]
        cs = [{"Container ID": "ta-smoke-orphan", "App ID": "ap-s", "App Name": "singularity-smoke", "Start Time": _iso(60)}]
        calls = []
        rep = ops.audit(mock_provider(apps, cs, calls), self.policy, terminate_orphans=True, now=NOW)
        self.assertEqual(rep["our_billable_containers"], 1)
        self.assertEqual(rep["foreign_containers"], 0)
        self.assertEqual([o["container_id"] for o in rep["orphans"]], ["ta-smoke-orphan"])
        self.assertIn(["container", "stop", "ta-smoke-orphan"], calls)


class MeasurementTests(unittest.TestCase):
    def test_gpu_snapshot_parses_and_tracks_peak(self):
        ops._PEAK["vram_mib"] = 0.0
        out = ops.gpu_snapshot(lambda: "0, 90, 1000, 183000\n1, 10, 5000, 183000\n")
        self.assertEqual(out["total_vram_used_mib"], 6000.0)
        self.assertEqual(out["mean_util_pct"], 50.0)
        self.assertEqual(out["peak_single_gpu_vram_mib_observed"], 5000.0)
        ops.gpu_snapshot(lambda: "0, 0, 100, 183000\n")
        self.assertEqual(ops._PEAK["vram_mib"], 5000.0)

    def test_gpu_snapshot_without_nvidia_smi(self):
        def boom():
            raise FileNotFoundError("nvidia-smi")

        self.assertFalse(ops.gpu_snapshot(boom)["available"])

    def test_cold_start_report_math(self):
        from deploy.modal.benchmark import RequestSample

        seq = iter([5000.0, 100.0, 120.0, 110.0])

        def sampler(url, prompt, max_tokens, **kw):
            return RequestSample(next(seq), 6000.0, 2, 8, [], "stop")

        rep = ops.measure_cold_start("http://x", sampler=sampler, warm_requests=3)
        self.assertEqual(rep["kind"], "measured")
        self.assertEqual(rep["warm_ttft_median_ms"], 110.0)
        self.assertEqual(rep["cold_overhead_ms"], 4890.0)

    def test_cold_start_validates_input(self):
        with self.assertRaises(ValueError):
            ops.measure_cold_start("http://x", warm_requests=-1, sampler=lambda *a, **k: None)

class WatchdogTests(unittest.TestCase):
    def _wd(self, tracker, drain=10.0, alive=None):
        self.calls = []
        return LifecycleWatchdog(
            tracker,
            drain_timeout_secs=drain,
            stop_child=lambda: self.calls.append("stop_child"),
            exit_fn=lambda: self.calls.append("exit"),
            liveness=alive,
        )

    def test_healthy_container_is_left_alone(self):
        clk = FakeClock()
        t = ready_tracker(clk, max_lifetime_secs=100)
        wd = self._wd(t)
        clk.advance(50)
        self.assertIsNone(wd.check())
        self.assertEqual(self.calls, [])

    def test_lifetime_stops_idle_container_without_any_request(self):
        clk = FakeClock()
        t = ready_tracker(clk, max_lifetime_secs=100)
        wd = self._wd(t)
        clk.advance(101)
        self.assertEqual(wd.check(), "lifetime")
        self.assertEqual(self.calls, ["stop_child", "exit"])
        self.assertEqual(t.state, LifecycleState.STOPPED)
        self.assertEqual(wd.check(), "lifetime")  # idempotent: no second exit
        self.assertEqual(self.calls, ["stop_child", "exit"])

    def test_spend_limit_stops_container(self):
        clk = FakeClock()
        t = ready_tracker(clk, gpu_count=8, gpu_usd_per_gpu_hour=10.0, max_spend_usd=40.0)
        wd = self._wd(t)
        clk.advance(1800)  # $40 at $80/h
        self.assertEqual(wd.check(), "spend")
        self.assertEqual(t.state, LifecycleState.STOPPED)
        self.assertEqual(self.calls[-1], "exit")

    def test_in_flight_work_gets_bounded_drain_then_is_abandoned(self):
        clk = FakeClock()
        t = ready_tracker(clk, max_lifetime_secs=100, request_timeout_secs=10_000)
        t.admit("a")
        wd = self._wd(t, drain=30.0)
        clk.advance(101)
        self.assertIsNone(wd.check())  # draining, job still running: do NOT kill yet
        self.assertEqual(t.state, LifecycleState.DRAINING)
        self.assertEqual(self.calls, [])
        with self.assertRaises(AdmissionRejected):
            t.admit("b")
        clk.advance(10)
        t.release("a")  # finishes inside the grace window -> clean exit
        self.assertEqual(wd.check(), "lifetime")
        self.assertEqual(t.failed, 0)
        self.assertEqual(self.calls, ["stop_child", "exit"])

    def test_stuck_job_is_abandoned_after_drain_timeout(self):
        clk = FakeClock()
        t = ready_tracker(clk, max_lifetime_secs=100, request_timeout_secs=10_000)
        t.admit("a")
        wd = self._wd(t, drain=30.0)
        clk.advance(101)
        self.assertIsNone(wd.check())
        clk.advance(31)
        self.assertEqual(wd.check(), "lifetime")
        self.assertEqual((t.in_flight, t.failed), (0, 1))
        self.assertEqual(t.state, LifecycleState.STOPPED)

    def test_dead_child_stops_container(self):
        clk = FakeClock()
        t = ready_tracker(clk)
        wd = self._wd(t, alive=lambda: False)
        self.assertEqual(wd.check(), "worker_failed")
        self.assertEqual(t.state, LifecycleState.STOPPED)
        self.assertEqual(self.calls, ["stop_child", "exit"])

    def test_exit_still_called_if_stop_child_raises(self):
        clk = FakeClock()
        t = ready_tracker(clk, max_lifetime_secs=1)
        calls = []

        def boom():
            raise RuntimeError("child wedged")

        wd = LifecycleWatchdog(
            t, drain_timeout_secs=1, stop_child=boom, exit_fn=lambda: calls.append("exit")
        )
        clk.advance(2)
        with self.assertRaises(RuntimeError):
            wd.check()
        self.assertEqual(calls, ["exit"])  # the GPUs must not be left billing

    def test_background_thread_exits_container(self):
        t = ready_tracker(_time.monotonic, max_lifetime_secs=0.05)
        done = threading.Event()
        wd = LifecycleWatchdog(
            t, drain_timeout_secs=1, stop_child=lambda: None, exit_fn=done.set, poll_secs=0.02
        )
        wd.start()
        self.assertTrue(done.wait(3.0))
        self.assertEqual(t.state, LifecycleState.STOPPED)

    def test_exit_container_helper_sends_sigint_and_arms_hard_exit(self):
        from unittest import mock

        from deploy.modal import app as app_mod

        with mock.patch.object(app_mod.os, "kill") as kill, mock.patch(
            "threading.Timer"
        ) as timer:
            app_mod._exit_container(grace_secs=5)
        import signal

        kill.assert_called_once_with(app_mod.os.getpid(), signal.SIGINT)
        timer.assert_called_once()
        self.assertEqual(timer.call_args[0][0], 5)
        timer.return_value.start.assert_called_once()

    def test_hard_exit_runs_cleanup_first_and_exits_nonzero(self):
        from unittest import mock

        from deploy.modal import app as app_mod

        order = []
        with mock.patch.object(app_mod.os, "_exit", side_effect=lambda c: order.append(("exit", c))):
            app_mod._hard_exit(cleanup=lambda: order.append("cleanup"))
        self.assertEqual(order, ["cleanup", ("exit", app_mod.HARD_EXIT_CODE)])
        self.assertNotEqual(app_mod.HARD_EXIT_CODE, 0)

    def test_hard_exit_survives_failing_and_hanging_cleanup(self):
        from unittest import mock

        from deploy.modal import app as app_mod

        codes = []
        with mock.patch.object(app_mod.os, "_exit", side_effect=codes.append):
            app_mod._hard_exit(cleanup=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
            self.assertEqual(codes, [app_mod.HARD_EXIT_CODE])
            codes.clear()
            release = threading.Event()
            self.addCleanup(release.set)
            with mock.patch.object(app_mod, "HARD_EXIT_CLEANUP_SECS", 0.2):
                t0 = _time.monotonic()
                app_mod._hard_exit(cleanup=lambda: release.wait(30))
            self.assertLess(_time.monotonic() - t0, 5)
            self.assertEqual(codes, [app_mod.HARD_EXIT_CODE])

    def test_exit_container_timer_callback_is_the_hard_exit_with_cleanup(self):
        from unittest import mock

        from deploy.modal import app as app_mod

        cleanup = lambda: None  # noqa: E731
        with mock.patch.object(app_mod.os, "kill"), mock.patch("threading.Timer") as timer:
            app_mod._exit_container(grace_secs=5, cleanup=cleanup)
        with mock.patch.object(app_mod, "_hard_exit") as hard:
            timer.call_args[0][1]()
        hard.assert_called_once_with(cleanup)

    def test_lifetime_clock_starts_at_ready_not_at_container_start(self):
        clock = FakeClock()
        t = LifecycleTracker(max_lifetime_secs=100, clock=clock)
        t.transition(LifecycleState.INITIALIZING)
        clock.advance(600)  # slow model load, far beyond the 100 s serving lifetime
        self.assertIsNone(t.limit_reason())
        self.assertEqual(t.serving_uptime(), 0.0)
        t.transition(LifecycleState.READY)
        clock.advance(99)
        self.assertIsNone(t.limit_reason())
        clock.advance(2)
        self.assertEqual(t.limit_reason(), "lifetime")
        self.assertAlmostEqual(t.snapshot()["serving_uptime_secs"], 101, places=1)

    def test_spend_guard_still_counts_from_container_start(self):
        clock = FakeClock()
        t = LifecycleTracker(
            max_lifetime_secs=10_000,
            gpu_count=8,
            gpu_usd_per_gpu_hour=3600.0 / 8,  # $1 per second in total
            max_spend_usd=50,
            clock=clock,
        )
        t.transition(LifecycleState.INITIALIZING)
        clock.advance(60)  # billing started at container start, before READY
        t.transition(LifecycleState.READY)
        self.assertEqual(t.limit_reason(), "spend")


try:
    import fastapi  # noqa: F401
    import httpx  # noqa: F401

    HAVE_FASTAPI = True
except ImportError:
    HAVE_FASTAPI = False


@unittest.skipUnless(HAVE_FASTAPI, "fastapi/httpx not installed; gateway tests need them")
class GatewayTests(unittest.TestCase):
    """Runs the real ASGI gateway against a local fake upstream."""

    def _upstream(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')

        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    def test_admission_status_and_release(self):
        from fastapi.testclient import TestClient

        from deploy.modal.app import create_asgi_proxy_app

        tracker = ready_tracker(clock=__import__("time").monotonic, max_concurrent=1)
        app = create_asgi_proxy_app(self._upstream(), 30, tracker=tracker, liveness=lambda: True)
        with TestClient(app) as c:
            self.assertEqual(c.post("/v1/chat/completions", json={}).status_code, 200)
            self.assertEqual(tracker.in_flight, 0)
            self.assertEqual(tracker.completed, 1)
            snap = c.get("/__lifecycle/status").json()
            self.assertEqual(snap["state"], "ready")
            tracker.begin_drain()
            r = c.post("/v1/chat/completions", json={})
            self.assertEqual((r.status_code, r.json()["code"]), (503, "draining"))
            self.assertEqual(c.get("/health").status_code, 200)  # probes unaffected by drain

    def test_dead_worker_rejected(self):
        from fastapi.testclient import TestClient

        from deploy.modal.app import create_asgi_proxy_app

        tracker = ready_tracker(clock=__import__("time").monotonic)
        app = create_asgi_proxy_app(self._upstream(), 30, tracker=tracker, liveness=lambda: False)
        with TestClient(app) as c:
            r = c.post("/x", json={})
            self.assertEqual((r.status_code, r.json()["code"]), (503, "worker_dead"))
            self.assertEqual(tracker.state, LifecycleState.FAILED)

    def _stalling_upstream(self, stall_secs: float, *, send_headers: bool):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if send_headers:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", "100")
                    self.end_headers()
                    self.wfile.write(b"{")
                    self.wfile.flush()
                _time.sleep(stall_secs)

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        self.addCleanup(srv.server_close)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    def test_504_timeout_closes_upstream_response(self):
        import httpx
        from fastapi.testclient import TestClient
        from unittest import mock

        from deploy.modal.app import create_asgi_proxy_app

        seen = []
        real_send = httpx.AsyncClient.send

        async def spy(self_, request, **kw):
            resp = await real_send(self_, request, **kw)
            seen.append(resp)
            return resp

        tracker = ready_tracker(clock=_time.monotonic)
        url = self._stalling_upstream(5.0, send_headers=True)
        app = create_asgi_proxy_app(url, 0.5, tracker=tracker, liveness=lambda: True)
        with mock.patch.object(httpx.AsyncClient, "send", spy):
            with TestClient(app) as c:
                r = c.post("/v1/chat/completions", json={})
        self.assertEqual((r.status_code, r.json()["code"]), (504, "timeout"))
        self.assertEqual(len(seen), 1, "upstream headers were sent, so a response must exist")
        self.assertTrue(seen[0].is_closed, "upstream httpx response leaked on 504 path")
        self.assertEqual((tracker.in_flight, tracker.failed), (0, 1))

    def test_504_while_waiting_for_headers_releases_slot(self):
        from fastapi.testclient import TestClient

        from deploy.modal.app import create_asgi_proxy_app

        tracker = ready_tracker(clock=_time.monotonic)
        url = self._stalling_upstream(5.0, send_headers=False)
        app = create_asgi_proxy_app(url, 0.5, tracker=tracker, liveness=lambda: True)
        with TestClient(app) as c:
            r = c.post("/x", json={})
        self.assertEqual(r.status_code, 504)
        self.assertEqual((tracker.in_flight, tracker.failed), (0, 1))

    def test_client_request_id_is_not_the_dedupe_key(self):
        from fastapi.testclient import TestClient

        from deploy.modal.app import create_asgi_proxy_app

        tracker = ready_tracker(clock=_time.monotonic)
        admitted = []
        real_admit = tracker.admit
        tracker.admit = lambda rid: (admitted.append(rid), real_admit(rid))[1]
        app = create_asgi_proxy_app(self._upstream(), 30, tracker=tracker, liveness=lambda: True)
        with TestClient(app) as c:
            for _ in range(2):
                r = c.post("/x", json={}, headers={"x-request-id": "victim"})
                self.assertEqual(r.status_code, 200)
        self.assertEqual(len(admitted), 2)
        self.assertNotIn("victim", admitted)
        self.assertEqual(len(set(admitted)), 2)
        self.assertEqual(tracker.rejected.get("duplicate", 0), 0)

    def _sse_upstream(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    self.wfile.write(b"data: {}\n\n")
                    self.wfile.flush()
                    _time.sleep(10)
                except OSError:
                    pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        self.addCleanup(srv.server_close)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    def _drive_never_read_stream(self, *, spec_version: str, send_raises: bool):
        import asyncio

        from deploy.modal.app import create_asgi_proxy_app

        tracker = ready_tracker(clock=_time.monotonic)
        # 30 s request deadline: a pass proves release did NOT wait for it.
        app = create_asgi_proxy_app(self._sse_upstream(), 30, tracker=tracker, liveness=lambda: True)

        async def run():
            sent_body = False

            async def receive():
                nonlocal sent_body
                if not sent_body:
                    sent_body = True
                    return {"type": "http.request", "body": b"{}", "more_body": False}
                await asyncio.sleep(0.3)  # then the client vanishes
                return {"type": "http.disconnect"}

            async def send(msg):
                if msg["type"] == "http.response.start":
                    if send_raises:
                        raise OSError("client gone")
                    await asyncio.sleep(3600)  # client never reads

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": spec_version},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/chat/completions",
                "raw_path": b"/v1/chat/completions",
                "query_string": b"",
                "root_path": "",
                "headers": [(b"content-type", b"application/json"), (b"content-length", b"2")],
                "client": ("127.0.0.1", 1),
                "server": ("127.0.0.1", 80),
            }
            try:
                await app(scope, receive, send)
            except BaseException:  # noqa: BLE001 - ClientDisconnect etc. are expected here
                pass

        t0 = _time.monotonic()
        asyncio.run(run())
        return tracker, _time.monotonic() - t0

    def test_never_iterated_stream_releases_slot_without_waiting_for_deadline(self):
        for spec, raises in (("2.3", False), ("2.4", True)):
            with self.subTest(spec=spec):
                tracker, elapsed = self._drive_never_read_stream(spec_version=spec, send_raises=raises)
                self.assertLess(elapsed, 10, "must not wait for the 30 s per-request deadline")
                self.assertEqual(tracker.in_flight, 0, "admission slot leaked")
                self.assertEqual(tracker.failed, 1)

    def test_sse_client_disconnect_after_done_counts_completed_and_tokens(self):
        """Standard SSE clients close the socket right after reading `data: [DONE]`.
        That must count as `completed` (not `failed`) and record `tokens_out`."""
        import asyncio
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        from deploy.modal.app import create_asgi_proxy_app

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    for piece in ("hel", "lo", " world"):
                        chunk = {"choices": [{"delta": {"content": piece}, "finish_reason": None}]}
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    done = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
                    self.wfile.write(f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n".encode())
                    self.wfile.flush()
                    _time.sleep(2.0)  # upstream stays open briefly while client disconnects
                except OSError:
                    pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        self.addCleanup(srv.server_close)
        url = f"http://127.0.0.1:{srv.server_address[1]}"

        tracker = ready_tracker(clock=_time.monotonic)
        app = create_asgi_proxy_app(url, 30, tracker=tracker, liveness=lambda: True)

        async def run():
            sent_body = False
            saw_done_Event = asyncio.Event()

            async def receive():
                nonlocal sent_body
                if not sent_body:
                    sent_body = True
                    return {"type": "http.request", "body": b"{}", "more_body": False}
                await saw_done_Event.wait()
                return {"type": "http.disconnect"}

            async def send(msg):
                if msg["type"] == "http.response.body" and b"[DONE]" in msg.get("body", b""):
                    saw_done_Event.set()

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/chat/completions",
                "raw_path": b"/v1/chat/completions",
                "query_string": b"",
                "root_path": "",
                "headers": [(b"content-type", b"application/json"), (b"content-length", b"2")],
                "client": ("127.0.0.1", 1),
                "server": ("127.0.0.1", 80),
            }
            await app(scope, receive, send)

        asyncio.run(run())
        self.assertEqual(tracker.in_flight, 0)
        self.assertEqual(tracker.completed, 1)
        self.assertEqual(tracker.failed, 0)
        self.assertEqual(tracker.tokens_out, 3)

    def test_non_streaming_json_records_completion_tokens(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from fastapi.testclient import TestClient

        from deploy.modal.app import create_asgi_proxy_app

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":7}}')

        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        url = f"http://127.0.0.1:{srv.server_address[1]}"

        tracker = ready_tracker(clock=_time.monotonic)
        app = create_asgi_proxy_app(url, 30, tracker=tracker, liveness=lambda: True)
        with TestClient(app) as c:
            r = c.post("/v1/chat/completions", json={})
            self.assertEqual(r.status_code, 200)
        self.assertEqual(tracker.completed, 1)
        self.assertEqual(tracker.tokens_out, 7)


class SnapshotLimitsTests(unittest.TestCase):
    def test_snapshot_exposes_effective_limits(self) -> None:
        t = LifecycleTracker(
            max_concurrent=2,
            request_timeout_secs=30,
            max_lifetime_secs=60,
            gpu_count=1,
            gpu_usd_per_gpu_hour=1.0,
            max_spend_usd=2.0,
        )
        self.assertEqual(
            t.snapshot()["limits"],
            {"max_lifetime_secs": 60, "max_spend_usd": 2.0, "request_timeout_secs": 30},
        )


if __name__ == "__main__":
    unittest.main()
