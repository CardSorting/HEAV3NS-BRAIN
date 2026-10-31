"""Unit tests for the scale-to-zero cost policy (no Modal/GPU required)."""

import sys
import unittest
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
_MODAL_DIR = str(Path(__file__).resolve().parent)
if _REPO_ROOT in sys.path and _MODAL_DIR in sys.path[: sys.path.index(_REPO_ROOT)]:
    sys.path.remove(_REPO_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_loaded_deploy = sys.modules.get("deploy")
if _loaded_deploy is not None and not hasattr(_loaded_deploy, "__path__"):
    del sys.modules["deploy"]

from deploy.modal import config  # noqa: E402
from deploy.modal.config import resolve_scaling_policy  # noqa: E402


class ScalingPolicyTests(unittest.TestCase):
    def test_default_is_scale_to_zero(self):
        p = resolve_scaling_policy({})
        self.assertEqual(p.min_containers, 0)
        self.assertFalse(p.pinned_warm)
        self.assertEqual(config.MIN_CONTAINERS, 0)
        self.assertLessEqual(p.scaledown_window_secs, config.SCALEDOWN_WINDOW_MAX_SECS)
        self.assertEqual(p.max_containers, 1)

    def test_warm_pin_refused_without_opt_in(self):
        with self.assertRaises(ValueError):
            resolve_scaling_policy({"SINGULARITY_MIN_CONTAINERS": "1"})

    def test_warm_pin_with_opt_in_is_flagged(self):
        p = resolve_scaling_policy(
            {"SINGULARITY_MIN_CONTAINERS": "1", "SINGULARITY_ALLOW_PINNED_WARM": "1"}
        )
        self.assertTrue(p.pinned_warm)

    def test_bounds(self):
        bad = [
            {"SINGULARITY_MIN_CONTAINERS": "-1"},
            {"SINGULARITY_MAX_CONTAINERS": "0"},
            {"SINGULARITY_MAX_CONTAINERS": "2"},
            {"SINGULARITY_SCALEDOWN_WINDOW_SECS": "0"},
            {"SINGULARITY_SCALEDOWN_WINDOW_SECS": "100000"},
            {"SINGULARITY_REQUEST_TIMEOUT_SECS": "0"},
            {"SINGULARITY_REQUEST_TIMEOUT_SECS": "999999"},
            {"SINGULARITY_SCALEDOWN_WINDOW_SECS": "abc"},
        ]
        for env in bad:
            with self.subTest(env=env), self.assertRaises(ValueError):
                resolve_scaling_policy(env)

    def test_overrides_within_bounds(self):
        p = resolve_scaling_policy(
            {"SINGULARITY_SCALEDOWN_WINDOW_SECS": "60", "SINGULARITY_REQUEST_TIMEOUT_SECS": "120"}
        )
        self.assertEqual((p.scaledown_window_secs, p.request_timeout_secs), (60, 120))


if __name__ == "__main__":
    unittest.main()


class PolicyEnvSnapshotTests(unittest.TestCase):
    def test_snapshot_is_whitelisted_and_skips_blank(self) -> None:
        from deploy.modal.config import policy_env_snapshot

        env = {
            "SINGULARITY_MAX_WORKER_LIFETIME_SECS": "60",
            "SINGULARITY_MAX_SPEND_USD": " ",
            "SINGULARITY_API_KEY": "secret",
            "HF_TOKEN": "secret",
        }
        self.assertEqual(policy_env_snapshot(env), {"SINGULARITY_MAX_WORKER_LIFETIME_SECS": "60"})
