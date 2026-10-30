# SINGULARITY bounded GPU validation

This app is a short-lived device and kernel test. It does **not** load V4.1, expose an API, benchmark inference, initialize NCCL, or deploy a persistent service. It requests exactly eight B200 GPUs in one container, verifies the CUDA-visible device names and SM 10.0 capability, and runs the Rust CUDA tests on device 0. The image is built on CPU first. The app has no warm replicas, no retries, one container maximum, a 300-second function timeout and single-use containers.

Prerequisites: Modal CLI 1.4.3, an authenticated workspace with eight B200 capacity and CUDA 13-compatible drivers, and permission to incur compute charges. The CUDA 13.0.2 Linux image is pinned by digest, as are Rust 1.88.0 and `Cargo.lock`; Ubuntu package repositories are not pinned to snapshots, so the image is not fully reproducible bit-for-bit. `cudarc` still uses CUDA 12.5 bindings, and its behavior with CUDA 13 requires the physical test. No production image has been established.

```bash
modal run deploy/modal/gpu_validation.py
# Optional: fetch verified expert ranges and run the real-weight matvec test
modal run deploy/modal/gpu_validation.py --real-weights
```

The local entrypoint saves successful test output and device inventory under `artifacts/b200-validation/`. Modal also retains logs under the run URL. For local checks without connecting to Modal, import the module with the Modal CLI's Python interpreter; importing does not create resources.

As published on 2026-10-30, Modal lists B200 at $0.001736 per GPU-second. Eight GPUs for 300 seconds are approximately $4.17 for GPU time, plus CPU, memory, image build, storage and any other applicable charges. Startup, queueing and actual billing can differ; this is not a spend cap. Check current rates and the workspace budget before running. Sources: https://modal.com/pricing and https://modal.com/docs/guide/budgets.

The persistent eight-B200 service described in `singularity2.md` remains unimplemented and must not be started using this validation app.
