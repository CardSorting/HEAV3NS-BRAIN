# Security Policy

## Supported Versions

The following versions of HEAV3NS-BRAIN (Project SINGULARITY) currently receive security updates:

| Version | Supported          |
| ------- | ------------------ |
| `main` branch | ✅ Active |
| Tagged releases | ✅ Latest only |
| Older tags | ❌ No longer supported |

---

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability, please report it via one of the following private channels:

1. **GitHub Private Security Advisory** (preferred):  
   Navigate to the [Security tab](https://github.com/CardSorting/HEAV3NS-BRAIN/security/advisories/new) and open a private advisory.

2. **Email:**  
   Send details to **willcruzdesigner@gmail.com** with the subject line:  
   `[SECURITY] HEAV3NS-BRAIN: <brief description>`

### What to include

Please provide as much of the following information as possible to help us triage and resolve the issue quickly:

- **Type of vulnerability** (e.g., memory safety, unsafe code path, denial of service, information disclosure)
- **Affected crate(s)** (e.g., `singularity-server`, `singularity-cuda`)
- **Steps to reproduce**
- **Proof of concept** or exploit code (if available)
- **Potential impact** and severity assessment
- **Suggested fix** (if you have one)

---

## Response Timeline

| Milestone | Target |
|---|---|
| Acknowledgement of report | Within **48 hours** |
| Initial triage and severity assessment | Within **5 business days** |
| Status update to reporter | Every **7 days** until resolved |
| Patch release (critical) | Within **14 days** of confirmation |
| Public disclosure | After patch is available and reporter is notified |

---

## Security Considerations for This Project

HEAV3NS-BRAIN is an inference runtime that handles:

- **GPU memory management** — block manager and KV cache operate with `unsafe` Rust for device memory access. All allocations are bounds-checked by the `BlockManager`.
- **Network-facing HTTP server** — `singularity-server` exposes REST/SSE endpoints. Rate limiting, circuit breaking, and admission control are implemented but should be reviewed before public deployment.
- **Model weight loading** — Safetensors files are loaded via `memmap2`. Ensure model files originate from trusted sources.
- **Shell / process boundaries** — The CUDA backend shells out to driver APIs. Ensure your CUDA environment is properly sandboxed.

### Deployment Hardening Recommendations

- Run `singularity-server` behind a reverse proxy (nginx, Caddy) with TLS termination.
- Never expose the dev-mock server (`--dev-mock`) in production.
- Restrict the `/metrics` endpoint to internal networks.
- Validate all Safetensors model files with checksums before loading.
- Use OS-level memory limits (`ulimit`) when running without NVIDIA MIG isolation.

---

## Out of Scope

The following are **not** considered security vulnerabilities for this project:

- Vulnerabilities in upstream Rust crates (report to the respective crate maintainers via [RustSec](https://rustsec.org/))
- Performance degradation without a security impact
- Issues only reproducible with root/administrator privileges on the host
- Social engineering attacks

---

## Acknowledgements

We thank all security researchers who responsibly disclose vulnerabilities. Contributors who report valid issues will be acknowledged in the release notes (with their permission).

