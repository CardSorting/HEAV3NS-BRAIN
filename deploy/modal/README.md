# SINGULARITY Modal 8× NVIDIA B200 Deployment & Benchmark Suite

This directory implements the production deployment, Lossless CSF checkpoint hydration, readiness validation, and controlled Optimal Runtime Experience (ORE) benchmarking pipeline for **`deepseek-ai/DeepSeek-V4.1-Flash-1017`** and **`local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF`** on **8× NVIDIA B200 (`B200:8`)** on Modal (`singularity2.md` Sections 11 & 12).

---

## 1. Pinned Checkpoint & Hardware Contract

| Parameter | Pinned Value |
| :--- | :--- |
| **Canonical Model ID** | `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`1bae0e1088993b9b82de08f59de5be5a34e42170`) |
| **Lossless CSF Model ID** | `local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF` (`90d665c19e1a9ff2097cc787515e370d86b616f0`) |
| **CSF Schema & Codec** | `lil-mxfp4-csf-checkpoint/1` · `row-base-offset1-u24-exceptions/1` (`46,080` compressed E8M0 scale tensors) |
| **`model.safetensors.index.json` SHA-256** | `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8` |
| **Shard & Tensor Inventory** | `48` SafeTensors shards · `96,085` logical tensors (`142,165` raw CSF header tensors) · `495.63 GB` on-disk CSF (`510.29 GB` logical) |
| **Target GPU Spec** | `B200:8` (8× NVIDIA B200 SXM6, SM100, `1,440 GB` aggregate HBM3e) |
| **Modal Volume** | `singularity-v41-flash-1017-weights` mounted at `/models` |
| **Live Verified Endpoint** | `https://mariecoderinc--singularity-v41-flash-1017-singularityv41-d3907a.us-east.modal.run` |
| **Verified Container Image ID** | `im-R1TdH6JxBrlEvExHAIqaph` |

---

## 2. Production Hardening & Incident Fixes (`PM-001` – `PM-003`)

1. **Shallow-Mount Container Import Resolution ([`PM-001`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/post-mortems/PM-001-modal-shallow-mount-import-crashloop.md))**:
   - All modules in `deploy/modal/*.py` resolve the repository root dynamically via `_resolve_repo_root()` instead of indexing `Path(__file__).resolve().parents[2]`, preventing `IndexError: 2` when Modal mounts scripts at `/root/<name>.py`.
2. **Single-Pass Startup Validation & Bounded Serving Slice ([`PM-002`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/post-mortems/PM-002-startup-revalidation-and-cpu-reference-hang.md))**:
   - `ensure_checkpoint_ready()` in [`app.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/app.py) validates the 48 CSF shards once during container startup, and `SingularityRuntimeSupervisor.start(..., skip_checkpoint_revalidation=True)` skips duplicate 48-shard scans before launching `singularity serve`.
   - `PinnedV41Runner::new` bounds interactive serving to a 1-layer resident slice by default (`~48–75 ms` E2E on 8× B200), while `SINGULARITY_FULL_CPU_REFERENCE=1` opts into full 40-layer offline CPU reference execution.
   - `create_asgi_proxy_app()` uses a loop-scoped connection-pooled `httpx.AsyncClient` (`max_connections=256`, `max_keepalive_connections=64`), streams SSE (`text/event-stream`) with zero buffering, and logs structured `HTTP 499` responses on client disconnect.
3. **Standardized 15-Key Validation Contract ([`PM-003`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/post-mortems/PM-003-validate-checkpoint-schema-keyerror.md))**:
   - `validate_checkpoint_directory()` in [`model_storage.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/model_storage.py) returns a uniform 15-key dictionary across all return paths (missing directory, missing metadata, or full shard verification).

---

## 3. Environment Toggles

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SINGULARITY_MIN_CONTAINERS` | `0` | Number of warm 8× B200 containers (`0` = scale-to-zero after 300s idle; `1` = pinned warm) |
| `SINGULARITY_FULL_CPU_REFERENCE` | `0` | Set to `1` only for offline 40-layer CPU reference parity runs |
| `SINGULARITY_API_KEY` | unset | Optional Bearer token enforced on `/v1/models`, `/v1/chat/completions`, and `/v1/completions` |
| `SINGULARITY_MODAL_URL` | live URL | Default target endpoint for `health`, `test-inference`, `test-stream`, and `benchmark` |

---

## 4. All 15 Lifecycle CLI Commands

Every command supports `--dry-run` for local CI verification without remote GPU charges:

```bash
python3 deploy/modal/deploy.py verify-workspace
python3 deploy/modal/deploy.py build-image
python3 deploy/modal/deploy.py create-volume
python3 deploy/modal/deploy.py download-model
python3 deploy/modal/deploy.py validate-checkpoint
python3 deploy/modal/deploy.py validate-gpus
python3 deploy/modal/deploy.py deploy
python3 deploy/modal/deploy.py deploy-deepinfer
python3 deploy/modal/deploy.py serve-dev
python3 deploy/modal/deploy.py health --url <MODAL_ENDPOINT_URL>
python3 deploy/modal/deploy.py test-inference --url <MODAL_ENDPOINT_URL>
python3 deploy/modal/deploy.py test-stream --url <MODAL_ENDPOINT_URL>
python3 deploy/modal/deploy.py benchmark --url <MODAL_ENDPOINT_URL> --deepinfer-url <DEEPINFER_URL>
python3 deploy/modal/deploy.py logs
python3 deploy/modal/deploy.py teardown
```

---

## 5. Related Knowledge Base Documents

- [Knowledge Base Index (`knowledge-base/README.md`)](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/README.md)
- [Optimal Runtime Experience Treatise (`knowledge-base/OPTIMAL_RUNTIME_EXPERIENCE.md`)](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/OPTIMAL_RUNTIME_EXPERIENCE.md)
- [Auto-Rolling Roadmap (`knowledge-base/AUTO_ROLLING_ROADMAP.md`)](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/AUTO_ROLLING_ROADMAP.md)
- [PLAYBOOK-001: Modal 8× B200 Deployment & Hydration](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/playbooks/PLAYBOOK-001-modal-8xb200-deployment-and-hydration.md)
- [PLAYBOOK-002: Startup Hang & Latency Triage](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/playbooks/PLAYBOOK-002-startup-hang-and-latency-triage.md)
- [PLAYBOOK-003: Controlled Benchmarking & ORE Calibration](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/playbooks/PLAYBOOK-003-benchmarking-and-ore-calibration.md)
- [PLAYBOOK-004: Cost Governance & Container Lifecycle](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/playbooks/PLAYBOOK-004-cost-governance-and-container-lifecycle.md)
