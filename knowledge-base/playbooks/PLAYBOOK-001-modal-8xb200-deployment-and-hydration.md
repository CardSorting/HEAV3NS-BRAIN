# PLAYBOOK-001: Modal 8× NVIDIA B200 Deployment & Lossless CSF Volume Hydration

**Target Environment**: Modal Serverless GPU Cloud (`B200:8`, Image `im-R1TdH6JxBrlEvExHAIqaph`)  
**Volume**: `singularity-v41-flash-1017-weights` mounted at `/models`  
**Canonical CLI**: [`deploy/modal/deploy.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/deploy.py)

---

## 1. Pre-Deployment Verification Checklist

Before triggering any remote build or 8× B200 deployment, run the local dry-run and unit test gates:

```bash
# 1. Verify all 15 lifecycle CLI commands in dry-run mode
python3 deploy/modal/deploy.py --dry-run verify-workspace
python3 deploy/modal/deploy.py --dry-run deploy

# 2. Run the Modal pipeline unit test suite (8 test suites)
python3 -m unittest discover -s deploy/modal -t . -p "test_*.py" -v

# 3. Run the Rust workspace test suite
cargo test --workspace
```

---

## 2. Step-by-Step Hydration & Deployment Procedure

### Step 1: Hydrate the Lossless CSF Checkpoint (`~495.63 GB`) onto the Modal Volume
Downloads `local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF` (`90d665c19e1a9ff2097cc787515e370d86b616f0`) on a CPU-only Modal worker (no GPU cost during download) and links `metadata/` and `tensors/` into `/models/DeepSeek-V4.1-Flash-1017`:

```bash
python3 deploy/modal/deploy.py download-model
```

### Step 2: Verify the Hydrated Volume Inventory
Confirms all 48 SafeTensors shards, `96,085` logical tensors (`46,080` CSF scale tensors, `142,165` raw header tensors), and `PINNED_INDEX_SHA256` (`74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`):

```bash
python3 deploy/modal/deploy.py validate-checkpoint
```

### Step 3: Validate the 8× NVIDIA B200 Topology & NVLink P2P Matrix
Verifies `8× NVIDIA B200` (`compute_capability = 10.0`), NVLink P2P matrix, and `bf16` matmul sanity:

```bash
python3 deploy/modal/deploy.py validate-gpus
```

### Step 4: Deploy the SINGULARITY 8× B200 Service
Builds (or reuses cached image `im-R1TdH6JxBrlEvExHAIqaph`) and deploys `singularity-v41-flash-1017`:

```bash
python3 deploy/modal/deploy.py deploy
```

### Step 5: Verify Live Readiness & Smoke Inference
```bash
export SINGULARITY_MODAL_URL="https://mariecoderinc--singularity-v41-flash-1017-singularityv41-d3907a.us-east.modal.run"

python3 deploy/modal/deploy.py health --url "$SINGULARITY_MODAL_URL"
python3 deploy/modal/deploy.py test-inference --url "$SINGULARITY_MODAL_URL"
```

