# Contributing to HEAV3NS-BRAIN (Project SINGULARITY)

Thank you for your interest in contributing! This document outlines the process for contributing code, documentation, and bug reports to Project SINGULARITY.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Commit Messages](#commit-messages)
- [Testing](#testing)
- [Crate Structure](#crate-structure)

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold it. Please report unacceptable behavior to the maintainers.

---

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/HEAV3NS-BRAIN.git
   cd HEAV3NS-BRAIN
   ```
3. **Add the upstream remote:**
   ```bash
   git remote add upstream https://github.com/CardSorting/HEAV3NS-BRAIN.git
   ```
4. **Create a feature branch:**
   ```bash
   git checkout -b feat/your-feature-name
   ```

---

## How to Contribute

### Reporting Bugs

Before opening an issue, please:

- Search [existing issues](https://github.com/CardSorting/HEAV3NS-BRAIN/issues) to avoid duplicates.
- Include the following in your report:
  - **Rust toolchain version** (`rustc --version`)
  - **CUDA Toolkit version** (if applicable)
  - **Hardware** (GPU model, compute capability)
  - **Steps to reproduce**
  - **Expected vs. actual behavior**
  - **Relevant log output / panic traces**

### Suggesting Features

Open a [GitHub Discussion](https://github.com/CardSorting/HEAV3NS-BRAIN/discussions) or issue with the label `enhancement`. Include:

- The **problem** you are trying to solve.
- The **proposed solution** with any relevant prior art (papers, other systems).
- Hardware impact (does it affect Blackwell SM100 / Hopper SM90 paths differently?).

### Pull Requests

1. Ensure your branch is rebased on the latest `main`:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```
2. All **61+ tests must pass**:
   ```bash
   cargo test --all
   ```
3. Run `clippy` with no errors:
   ```bash
   cargo clippy --all -- -D warnings
   ```
4. Format your code:
   ```bash
   cargo fmt --all
   ```
5. Open a PR against `main` with a clear description of **what** changed and **why**.
6. Link any related issues in the PR description (`Closes #123`).

PRs that add new inference subsystems should include corresponding integration tests in `singularity-tests/`.

---

## Development Setup

### Minimum Requirements

| Tool | Version |
|---|---|
| Rust (stable) | 1.88.0+ |
| Cargo | bundled with Rust |
| CUDA Toolkit _(optional)_ | 12.2+ |
| NVIDIA Driver _(optional)_ | 535+ |
| Python _(optional, fixture gen)_ | 3.11+ |

### Build (CPU mock — no GPU required)

```bash
cargo build --release
```

### Build (real CUDA hardware — H200 / B200)

```bash
cargo build --release --features cuda
```

### Run the test suite

```bash
# All 61+ tests
cargo test --all

# A specific crate
cargo test -p singularity-scheduler

# A specific test
cargo test -p singularity-model v41_quant
```

### Inspect your hardware

```bash
cargo run --bin singularity -- inspect-hardware
```

---

## Code Style

- Follow **idiomatic Rust**. Prefer `?` for error propagation over `unwrap()`.
- All public items must have **doc comments** (`///`).
- Keep unsafe blocks **minimal and documented** with a `// SAFETY:` comment.
- GPU-specific code paths must be gated by the `cuda` feature flag and have a corresponding mock/CPU fallback.
- Avoid `println!` in library code; use `tracing::{info, debug, warn, error}`.

---

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <short summary>

[optional body]

[optional footer(s)]
```

**Types:** `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`, `ci`

**Scopes** (crate names): `core`, `cuda`, `cache`, `scheduler`, `model`, `grammar`, `server`, `cli`, `tests`

**Examples:**
```
feat(scheduler): add ZoS deferred release across forward boundary
fix(cache): correct SHA256 key collision in radix tree eviction
perf(model): fuse RoPE rotations with attention projection on SM100
docs(server): document ring-buffer replay Last-Event-ID semantics
```

---

## Testing

| Test suite | Location | Coverage |
|---|---|---|
| Unit tests | Inside each `src/*.rs` file | Per-module logic |
| Integration tests | `singularity-tests/tests/` | Cross-crate |
| E2E server test | `tests/e2e_server.rs` | Full HTTP lifecycle |
| Failure modes | `tests/failure_modes.rs` | 18 edge cases |
| Real inference | `tests/real_inference.rs` | SmolLM2, DeepSeek-V2-Lite |
| Benchmarks | `singularity-tests/benches/` | Criterion microbenchmarks |

All new features require at least one accompanying test. Performance-sensitive paths should include a Criterion benchmark.

---

## Crate Structure

| Crate | Responsibility |
|---|---|
| `singularity-core` | Shared types, `SingularityError`, metrics, config |
| `singularity-cuda` | `GpuBackend` trait, hardware detection, CUDA kernels |
| `singularity-cache` | Paged block manager, radix prefix tree, FP4/MXFP8 layouts |
| `singularity-scheduler` | ZOS pipeline, LCFS preemption, DP consensus, MTP estimator |
| `singularity-model` | Safetensors loader, MLA attention, MoE EPLB, tokenizer |
| `singularity-grammar` | JSON bitmask compiler, KMP tool-call trigger matcher |
| `singularity-server` | Axum REST/SSE, admission, circuit breaker, ring buffer |
| `singularity-cli` | Developer CLI (`serve`, `benchmark`, `inspect-hardware`) |
| `singularity-tests` | E2E harness, Criterion benchmarks |

When adding a new crate, register it in the root `Cargo.toml` workspace `members` array.

