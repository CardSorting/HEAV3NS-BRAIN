# PLAYBOOK-004: 8× NVIDIA B200 Cost Governance & Container Lifecycle Management

**Applies To**: `deploy/modal/config.py`, `deploy/modal/app.py`, `deploy/modal/deploy.py`

---

## 1. Cost Profile of 8× NVIDIA B200 (`B200:8`)

An 8× NVIDIA B200 node on Modal (`1,440 GB` aggregate HBM3e) costs approximately **`$49.99/hour`** (`~$1,200/day`) when a container is warm/active.

To prevent accidental idle GPU burn while preserving instant scale-up capability:
- `MIN_CONTAINERS` in [`deploy/modal/config.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/config.py) defaults to **`0`** (`int(os.environ.get("SINGULARITY_MIN_CONTAINERS", "0"))`).
- `SCALEDOWN_WINDOW_SECS` is set to **`300` seconds (5 minutes)** so containers automatically scale to zero 5 minutes after the last HTTP request completes.
- Persistent model weights (`495.63 GB` Lossless CSF shards) reside on the `singularity-v41-flash-1017-weights` Modal Volume, which incurs zero GPU compute cost while idle.

---

## 2. Operational Modes

### Mode A: Scale-to-Zero On-Demand Mode (Default — `$0.00/hr` Idle)
```bash
SINGULARITY_MIN_CONTAINERS=0 python3 deploy/modal/deploy.py deploy
```
- **Behavior**: 0 warm containers when idle; scales `0 -> 1` automatically on the first incoming HTTP request (`~40–60s` cold start with single-pass CSF validation) and scales `1 -> 0` after 300s of inactivity.

### Mode B: Pinned Warm Production Mode (Zero Cold Start)
```bash
SINGULARITY_MIN_CONTAINERS=1 python3 deploy/modal/deploy.py deploy
```
- **Behavior**: Keeps 1× 8-GPU B200 container warm continuously (`TTFT P50 ~ 49ms` on every request).
- **Mandatory Rule**: Always revert to `SINGULARITY_MIN_CONTAINERS=0` or run `teardown` immediately after completing a benchmark session or live demo.

### Mode C: Immediate Teardown (Stop All Active Containers Instantly)
```bash
python3 deploy/modal/deploy.py teardown
```
- **Behavior**: Stops the `singularity-v41-flash-1017` and `deepinfer-v41-baseline` Modal apps immediately while preserving the hydrated `singularity-v41-flash-1017-weights` Volume intact.

