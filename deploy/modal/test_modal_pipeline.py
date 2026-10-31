"""Unit tests for the SINGULARITY Modal deployment, storage, health, and benchmark pipeline."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

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


_MODAL_DIR = str(Path(__file__).resolve().parent)
_REPO_ROOT = str(_resolve_repo_root(Path(__file__)))
if _MODAL_DIR in sys.path and (Path(_MODAL_DIR) / "deploy.py").is_file():
    sys.path.remove(_MODAL_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deploy.modal.app import (
    STARTUP_SEQUENCE_STEPS,
    build_singularity_serve_command,
    create_asgi_proxy_app,
    ensure_checkpoint_ready,
)
from deploy.modal.benchmark import (
    RequestSample,
    percentile,
    run_benchmark_suite,
    summarize_samples,
)
from deploy.modal.config import (
    CSF_CODEC,
    CSF_MODEL_ID,
    CSF_PINNED_REVISION,
    CSF_SCHEMA,
    DeploymentConfig,
    GPU_COUNT,
    GPU_SPEC,
    MODEL_ID,
    PINNED_INDEX_SHA256,
    PINNED_REVISION,
    PINNED_SHARD_COUNT,
    PINNED_TENSOR_COUNT,
    PINNED_TOTAL_BYTES,
    expected_shard_filenames,
)
from deploy.modal.deepinfer_app import (
    create_deepinfer_adapter_app,
    get_deepinfer_reference_spec,
)
from deploy.modal.deploy import ALL_LIFECYCLE_COMMANDS, execute_command
from deploy.modal.health import REQUIRED_READINESS_KEYS, check_service_health, run_smoke_inference
from deploy.modal.image import validate_workspace_sources
from deploy.modal.model_storage import (
    decode_mxfp4_csf_scale,
    inspect_safetensors_header,
    prepare_csf_checkpoint_directory,
    reconstruct_csf_shard,
    validate_checkpoint_directory,
)


def _write_safetensors_shard(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header = {}
    payload = bytearray()
    for name, (dtype, shape, raw_bytes) in tensors.items():
        start = len(payload)
        payload.extend(raw_bytes)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    header_bytes = json.dumps(header).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(payload)


class _MockSingularityHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            body = {"status": "healthy", "ready": True, "circuit_breaker": "Closed"}
            self._send_json(200, body)
        elif self.path == "/ready":
            checks = {k: True for k in REQUIRED_READINESS_KEYS}
            body = {"status": "ready", "ready": True, "model": MODEL_ID, **checks, "checks": checks}
            self._send_json(200, body)
        elif self.path == "/v1/models":
            body = {
                "object": "list",
                "data": [
                    {"id": "singularity-host-development", "object": "model"},
                    {"id": MODEL_ID, "object": "model"},
                ],
            }
            self._send_json(200, body)
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(length).decode("utf-8"))
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in ["Hello", " world"]:
                chunk = {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            done_chunk = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            self.wfile.write(f"data: {json.dumps(done_chunk)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            body = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": req.get("model", MODEL_ID),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello world"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            }
            self._send_json(200, body)

    def _send_json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class ModalPipelineTests(unittest.TestCase):
    def test_pinned_config_and_workspace_sources(self) -> None:
        cfg = DeploymentConfig()
        self.assertEqual(cfg.model_id, "deepseek-ai/DeepSeek-V4.1-Flash-1017")
        self.assertEqual(cfg.revision, PINNED_REVISION)
        self.assertEqual(cfg.csf_model_id, CSF_MODEL_ID)
        self.assertEqual(cfg.csf_revision, CSF_PINNED_REVISION)
        self.assertEqual(cfg.csf_schema, CSF_SCHEMA)
        self.assertEqual(cfg.csf_codec, CSF_CODEC)
        self.assertEqual(cfg.index_sha256, PINNED_INDEX_SHA256)
        self.assertEqual(cfg.shard_count, PINNED_SHARD_COUNT)
        self.assertEqual(cfg.tensor_count, PINNED_TENSOR_COUNT)
        self.assertEqual(cfg.total_bytes, PINNED_TOTAL_BYTES)
        self.assertEqual(cfg.gpu_spec, GPU_SPEC)
        self.assertEqual(cfg.gpu_count, GPU_COUNT)
        self.assertEqual(len(expected_shard_filenames()), 48)
        self.assertEqual(len(STARTUP_SEQUENCE_STEPS), 17)

        ws = validate_workspace_sources()
        self.assertTrue(ws["valid"], f"Missing workspace files: {ws['missing']}")

        cmd = build_singularity_serve_command(api_key="test-key")
        self.assertIn("--v41-config", cmd)
        self.assertIn("--use-cuda", cmd)
        self.assertIn("--dspark", cmd)
        self.assertIn("test-key", cmd)

        di = get_deepinfer_reference_spec()
        self.assertEqual(di["gpu_spec"], "B200:8")

    def test_safetensors_header_and_checkpoint_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps({"model_type": "deepseek_v41"}), encoding="utf-8"
            )
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            shard_name = "model-00001-of-00001.safetensors"
            _write_safetensors_shard(
                root / shard_name,
                {"embed.weight": ("BF16", [2, 2], b"\x00\x3f" * 4)},
            )
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 8},
                        "weight_map": {"embed.weight": shard_name},
                    }
                ),
                encoding="utf-8",
            )

            info = inspect_safetensors_header(root / shard_name)
            self.assertEqual(info["tensor_count"], 1)

            non_strict = validate_checkpoint_directory(
                root, strict_pinned=False, verify_shard_headers=True
            )
            self.assertTrue(non_strict["valid"], f"Unexpected errors: {non_strict['errors']}")
            self.assertEqual(non_strict["shard_count"], 1)
            self.assertEqual(non_strict["present_shards"], 1)
            self.assertGreater(non_strict["total_bytes"], 0)
            self.assertEqual(non_strict["total_bytes"], non_strict["total_shard_file_bytes"])

            reloaded = []
            ready_check = ensure_checkpoint_ready(
                str(root),
                reload_callback=lambda: reloaded.append(True),
                strict_pinned=False,
            )
            self.assertEqual(reloaded, [True])
            self.assertTrue(ready_check["valid"])
            self.assertEqual(ready_check["shard_count"], 1)
            self.assertGreater(ready_check["total_bytes"], 0)

            missing_dir = validate_checkpoint_directory(
                root / "nonexistent", strict_pinned=False, verify_shard_headers=False
            )
            self.assertFalse(missing_dir["valid"])
            self.assertEqual(missing_dir["shard_count"], 0)
            self.assertEqual(missing_dir["total_bytes"], 0)
            self.assertEqual(set(missing_dir.keys()), set(non_strict.keys()))

            empty_dir = root / "empty_ckpt"
            empty_dir.mkdir()
            missing_meta = validate_checkpoint_directory(
                empty_dir, strict_pinned=False, verify_shard_headers=False
            )
            self.assertFalse(missing_meta["valid"])
            self.assertEqual(set(missing_meta.keys()), set(non_strict.keys()))

            # Test ensure_checkpoint_ready download path + post-download validation
            import unittest.mock as mock

            dl_dir = root / "downloaded_ckpt"
            committed = []

            def _fake_download(target_dir, **kwargs):
                p = Path(target_dir)
                p.mkdir(parents=True, exist_ok=True)
                (p / "config.json").write_text(
                    json.dumps({"model_type": "deepseek_v41"}), encoding="utf-8"
                )
                (p / "tokenizer.json").write_text("{}", encoding="utf-8")
                _write_safetensors_shard(
                    p / shard_name,
                    {"embed.weight": ("BF16", [2, 2], b"\x00\x3f" * 4)},
                )
                (p / "model.safetensors.index.json").write_text(
                    json.dumps(
                        {
                            "metadata": {"total_size": 8},
                            "weight_map": {"embed.weight": shard_name},
                        }
                    ),
                    encoding="utf-8",
                )
                return {"status": "completed"}

            with mock.patch("deploy.modal.app.download_pinned_checkpoint", side_effect=_fake_download):
                dl_ready = ensure_checkpoint_ready(
                    str(dl_dir),
                    commit_callback=lambda: committed.append(True),
                    strict_pinned=False,
                )
                self.assertTrue(dl_ready["valid"])
                self.assertEqual(dl_ready["shard_count"], 1)
                self.assertGreater(dl_ready["total_bytes"], 0)
                self.assertEqual(committed, [True])

            # Test ensure_checkpoint_ready raises RuntimeError if validation still fails after download
            with mock.patch("deploy.modal.app.download_pinned_checkpoint", return_value={"status": "completed"}):
                with self.assertRaises(RuntimeError):
                    ensure_checkpoint_ready(
                        str(root / "still_empty"),
                        strict_pinned=False,
                    )

            strict = validate_checkpoint_directory(
                root, strict_pinned=True, verify_shard_headers=True
            )
            self.assertFalse(strict["valid"])
            self.assertEqual(strict["shard_count"], 1)
            self.assertGreater(strict["total_bytes"], 0)

    def test_csf_codec_reconstruct_and_checkpoint_preparation(self) -> None:
        import hashlib

        rows, cols = 18, 8
        expected = bytearray(rows * cols)
        fixed = bytearray()
        for b_start in range(0, rows, 16):
            blk_rows = min(16, rows - b_start)
            for r_in in range(blk_rows):
                fixed.append(120 + b_start + r_in)
            for r_in in range(blk_rows):
                r = b_start + r_in
                base = 120 + r
                mask = 0b10100101 if r % 2 == 0 else 0b01011010
                fixed.append(mask)
                for c in range(cols):
                    bit = (mask >> c) & 1
                    expected[r * cols + c] = base + bit

        exc0 = 3 | (135 << 24)
        exc1 = (17 * cols + 7) | (111 << 24)
        expected[3] = 135
        expected[17 * cols + 7] = 111
        exceptions = struct.pack("<II", exc0, exc1)

        decoded = decode_mxfp4_csf_scale(bytes(fixed), exceptions, rows, cols)
        self.assertEqual(decoded, bytes(expected))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta_dir = root / "metadata"
            tensors_dir = root / "tensors"
            receipts_dir = root / "receipts"
            meta_dir.mkdir()
            tensors_dir.mkdir()
            receipts_dir.mkdir()

            (meta_dir / "config.json").write_text(
                json.dumps({"model_type": "deepseek_v41"}), encoding="utf-8"
            )
            (meta_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            shard_name = "model-00001-of-00001.safetensors"
            weight_bytes = b"\x21" * (rows * cols * 16)
            w_end = len(weight_bytes)
            f_end = w_end + len(fixed)
            e_end = f_end + len(exceptions)
            csf_header = {
                "__metadata__": {"schema": CSF_SCHEMA, "codec": CSF_CODEC},
                "layers.0.ffn.experts.0.w1.weight": {
                    "dtype": "I8",
                    "shape": [rows, cols * 16],
                    "data_offsets": [0, w_end],
                },
                "layers.0.ffn.experts.0.w1.scale.mxfp4_csf_fixed": {
                    "dtype": "U8",
                    "shape": [len(fixed)],
                    "data_offsets": [w_end, f_end],
                },
                "layers.0.ffn.experts.0.w1.scale.mxfp4_csf_exceptions": {
                    "dtype": "U32",
                    "shape": [len(exceptions) // 4],
                    "data_offsets": [f_end, e_end],
                },
            }
            csf_header_bytes = json.dumps(csf_header).encode("utf-8")
            csf_shard_path = tensors_dir / shard_name
            with csf_shard_path.open("wb") as handle:
                handle.write(struct.pack("<Q", len(csf_header_bytes)))
                handle.write(csf_header_bytes)
                handle.write(weight_bytes)
                handle.write(fixed)
                handle.write(exceptions)

            (meta_dir / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": len(weight_bytes) + len(expected)},
                        "weight_map": {
                            "layers.0.ffn.experts.0.w1.weight": shard_name,
                            "layers.0.ffn.experts.0.w1.scale": shard_name,
                        },
                    }
                ),
                encoding="utf-8",
            )
            receipt_path = receipts_dir / f"{shard_name}.json"
            receipt_path.write_text(
                json.dumps(
                    {
                        "compressed_tensors": [
                            {
                                "name": "layers.0.ffn.experts.0.w1.scale",
                                "source_sha256": hashlib.sha256(expected).hexdigest(),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            prep = prepare_csf_checkpoint_directory(root)
            self.assertTrue(prep["is_csf_layout"])
            self.assertTrue((root / "config.json").is_file())
            self.assertTrue((root / "model.safetensors.index.json").is_file())

            validation = validate_checkpoint_directory(
                root, strict_pinned=False, verify_shard_headers=True
            )
            self.assertTrue(validation["valid"], f"Unexpected errors: {validation['errors']}")
            self.assertEqual(validation["verified_header_tensors"], 2)
            self.assertEqual(validation["verified_csf_scales"], 1)
            self.assertEqual(validation["schema"], CSF_SCHEMA)
            self.assertEqual(validation["codec"], CSF_CODEC)

            reconstructed_path = root / "reconstructed.safetensors"
            recon = reconstruct_csf_shard(
                csf_shard_path, reconstructed_path, receipt_path=receipt_path
            )
            self.assertEqual(recon["logical_tensors"], 2)
            self.assertEqual(recon["reconstructed_scales"], 1)
            self.assertEqual(recon["verified_receipts"], 1)
            recon_info = inspect_safetensors_header(reconstructed_path)
            self.assertEqual(recon_info["tensor_count"], 2)
            self.assertEqual(recon_info["csf_scale_count"], 0)

    def test_all_15_lifecycle_cli_commands_in_dry_run(self) -> None:
        self.assertEqual(len(ALL_LIFECYCLE_COMMANDS), 15)
        for subcmd in ALL_LIFECYCLE_COMMANDS:
            res = execute_command(["--dry-run", subcmd])
            self.assertIsInstance(res, dict)

    def test_health_smoke_and_benchmark_against_mock_http_server(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), _MockSingularityHandler)
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{port}"
            health = check_service_health(url)
            self.assertTrue(health["ready"], f"Health errors: {health['errors']}")

            smoke = run_smoke_inference(url)
            self.assertTrue(smoke["passed"], f"Smoke errors: {smoke['errors']}")
            self.assertIn("tool_thinking_chat", smoke["results"])

            csf_smoke = run_smoke_inference(url, model=CSF_MODEL_ID)
            self.assertTrue(csf_smoke["passed"], f"CSF smoke errors: {csf_smoke['errors']}")
            self.assertIn("tool_thinking_chat", csf_smoke["results"])

            with tempfile.TemporaryDirectory() as tmp:
                out_file = Path(tmp) / "bench.json"
                report = run_benchmark_suite(
                    url,
                    deepinfer_url=url,
                    configurations=["A_base", "D_scaling"],
                    tiers=["short_128"],
                    concurrencies=[1, 2],
                    max_tokens=4,
                    output_path=out_file,
                )
                self.assertTrue(report["correctness_gate"]["passed"])
                self.assertEqual(len(report["benchmarks"]), 3)
                self.assertEqual(len(report["deepinfer_benchmarks"]), 3)
                self.assertTrue(out_file.is_file())
        finally:
            server.shutdown()
            server.server_close()

    def test_script_mode_and_modal_cli_import_refs(self) -> None:
        import runpy

        repo_root = Path(_REPO_ROOT)
        for rel in ("deploy/modal/app.py", "deploy/modal/deepinfer_app.py"):
            ns = runpy.run_path(str(repo_root / rel))
            self.assertIn("CHECKPOINT_DIR", ns)

        # Simulate Modal container entrypoint where script is mounted at /root/<name>.py (only 2 parents)
        for rel, shallow_file in (
            ("deploy/modal/app.py", "/root/app.py"),
            ("deploy/modal/deepinfer_app.py", "/root/deepinfer_app.py"),
            ("deploy/modal/gpu_validation.py", "/root/gpu_validation.py"),
            ("deploy/modal/deploy.py", "/root/deploy.py"),
            ("deploy/modal/image.py", "/root/image.py"),
            ("deploy/modal/health.py", "/root/health.py"),
            ("deploy/modal/benchmark.py", "/root/benchmark.py"),
            ("deploy/modal/model_storage.py", "/root/model_storage.py"),
            ("deploy/modal/app.py", "/app.py"),
        ):
            import types

            code = (repo_root / rel).read_text(encoding="utf-8")
            temp_mod = types.ModuleType("_shallow_test_mod")
            temp_mod.__file__ = shallow_file
            temp_mod.__package__ = ""
            sys.modules["_shallow_test_mod"] = temp_mod
            try:
                exec(compile(code, shallow_file, "exec"), temp_mod.__dict__)
                self.assertIn("_resolve_repo_root", temp_mod.__dict__)
                resolved_root = temp_mod.__dict__["_resolve_repo_root"](Path(shallow_file))
                self.assertEqual(resolved_root, repo_root)
            finally:
                sys.modules.pop("_shallow_test_mod", None)

        try:
            import modal
            from modal.cli.import_refs import import_file_or_module, parse_import_ref
        except ImportError:
            modal = None

        if modal is not None:
            for ref_str in (
                "deploy/modal/app.py",
                "deploy/modal/deepinfer_app.py",
                "deploy/modal/gpu_validation.py",
            ):
                mod = import_file_or_module(parse_import_ref(str(repo_root / ref_str)))
                self.assertIsNotNone(mod)

    def test_percentile_and_summary_math(self) -> None:
        self.assertEqual(percentile([], 50), 0.0)
        self.assertAlmostEqual(percentile([10.0, 20.0, 30.0], 50), 20.0)
        sample = RequestSample(
            ttft_ms=12.0,
            total_latency_ms=42.0,
            prompt_tokens=16,
            completion_tokens=4,
            itl_ms_list=[10.0, 10.0, 10.0],
            finish_reason="stop",
        )
        self.assertAlmostEqual(sample.tpot_ms, 10.0)
        summary = summarize_samples([sample], 0.1, "B_optimized", "short_128", 1)
        self.assertEqual(summary["total_completion_tokens"], 4)
        self.assertAlmostEqual(summary["ttft_jitter_ratio"], 1.0)
        self.assertAlmostEqual(summary["itl_jitter_ratio"], 1.0)
        ore = summary["optimal_runtime_experience"]
        self.assertAlmostEqual(ore["balance_score"], 1.0)
        self.assertEqual(ore["stability_tier"], "optimal")

    def test_create_asgi_proxy_app_json_sse_and_upstream_failure(self) -> None:
        import asyncio

        try:
            import fastapi  # noqa: F401
            import httpx
        except ImportError:
            self.skipTest("fastapi/httpx not installed in current Python interpreter")

        server = HTTPServer(("127.0.0.1", 0), _MockSingularityHandler)
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            proxy = create_asgi_proxy_app(f"http://127.0.0.1:{port}", timeout_secs=5.0)
            dead_proxy = create_asgi_proxy_app("http://127.0.0.1:1", timeout_secs=1.0)

            async def _exercise() -> None:
                transport = httpx.ASGITransport(app=proxy)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://testserver"
                ) as client:
                    # 1. JSON response forwarding
                    r_health = await client.get("/health")
                    self.assertEqual(r_health.status_code, 200)
                    self.assertEqual(r_health.json()["status"], "healthy")
                    shared_client_1 = proxy.state.get_upstream_client()

                    # 2. SSE streaming forwarding (reuses pooled AsyncClient)
                    r_sse = await client.post(
                        "/v1/chat/completions",
                        json={"model": MODEL_ID, "messages": [], "stream": True},
                    )
                    self.assertEqual(r_sse.status_code, 200)
                    self.assertIn("text/event-stream", r_sse.headers.get("content-type", ""))
                    self.assertIn("[DONE]", r_sse.text)
                    self.assertIs(proxy.state.get_upstream_client(), shared_client_1)

                # 3. Unreachable upstream returns 502 Bad Gateway cleanly
                dead_transport = httpx.ASGITransport(app=dead_proxy)
                async with httpx.AsyncClient(
                    transport=dead_transport, base_url="http://testserver"
                ) as dead_client:
                    r_err = await dead_client.get("/health")
                    self.assertEqual(r_err.status_code, 502)
                    self.assertIn("Upstream connection failed", r_err.json()["error"])

                # 4. DeepInfer adapter app returns 503 on /ready when llm is None, and 200 when loaded
                class _DummyService:
                    ready = True
                    llm = None
                    tokenizer = None

                di_service = _DummyService()
                di_app = create_deepinfer_adapter_app(di_service)
                di_transport = httpx.ASGITransport(app=di_app)
                async with httpx.AsyncClient(
                    transport=di_transport, base_url="http://testserver"
                ) as di_client:
                    r_unready = await di_client.get("/ready")
                    self.assertEqual(r_unready.status_code, 503)
                    self.assertFalse(r_unready.json()["ready"])

                    class _DummyOut:
                        token_ids = [10, 20]

                    class _DummyReqOut:
                        outputs = [_DummyOut()]

                    class _DummyLlm:
                        def generate(self, prompts, sampling_params=None):
                            return [_DummyReqOut()]

                    class _DummyTokenizer:
                        def encode(self, text):
                            return [1, 2, 3]

                        def decode(self, ids, skip_special_tokens=True):
                            return "ok"

                    di_service.llm = _DummyLlm()
                    di_service.tokenizer = _DummyTokenizer()
                    r_ready = await di_client.get("/ready")
                    self.assertEqual(r_ready.status_code, 200)
                    self.assertTrue(r_ready.json()["ready"])

                    r_chat = await di_client.post(
                        "/v1/chat/completions",
                        json={
                            "model": MODEL_ID,
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 4,
                            "stream": False,
                        },
                    )
                    self.assertEqual(r_chat.status_code, 200)
                    self.assertEqual(r_chat.json()["choices"][0]["message"]["content"], "ok")

                # 5. ClientDisconnect while reading request body returns HTTP 499 cleanly without crashing
                sent_responses = []

                async def _disconnect_receive():
                    return {"type": "http.disconnect"}

                async def _capture_send(message):
                    sent_responses.append(message)

                scope = {
                    "type": "http",
                    "asgi": {"version": "3.0"},
                    "http_version": "1.1",
                    "method": "POST",
                    "scheme": "http",
                    "path": "/v1/chat/completions",
                    "raw_path": b"/v1/chat/completions",
                    "query_string": b"",
                    "headers": [(b"content-type", b"application/json")],
                    "client": ("127.0.0.1", 12345),
                    "server": ("testserver", 80),
                }
                await proxy(scope, _disconnect_receive, _capture_send)
                start_msg = next(m for m in sent_responses if m["type"] == "http.response.start")
                self.assertEqual(start_msg["status"], 499)
                await proxy.state.close_upstream_client()
                await dead_proxy.state.close_upstream_client()

            asyncio.run(_exercise())

            # 6. run_smoke_inference against unreachable port returns passed=False with HTTP 599 instead of raising
            dead_smoke = run_smoke_inference("http://127.0.0.1:1", timeout=0.5)
            self.assertFalse(dead_smoke["passed"])
            self.assertEqual(dead_smoke["results"]["chat"]["status"], 599)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()

