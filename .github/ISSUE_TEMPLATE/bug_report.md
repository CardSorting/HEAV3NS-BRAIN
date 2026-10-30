---
name: Bug Report
about: Report a reproducible bug or unexpected behavior
title: "[BUG] "
labels: bug
assignees: CardSorting
---

## Describe the Bug

A clear and concise description of what the bug is.

## Steps to Reproduce

```bash
# Exact commands to reproduce the behavior
cargo run --bin singularity -- serve ...
```

1. Step one
2. Step two
3. See error

## Expected Behavior

What did you expect to happen?

## Actual Behavior

What actually happened? Paste the full panic/error output below.

```
<paste error or panic trace here>
```

## Environment

| Field | Value |
|---|---|
| OS | e.g. Ubuntu 22.04 |
| Rust version (`rustc --version`) | |
| CUDA Toolkit version | e.g. 12.5 / N/A |
| NVIDIA Driver version | e.g. 555.42 / N/A |
| GPU model | e.g. H200 SXM / B200 / N/A (CPU mock) |
| Feature flags | e.g. `--features cuda` / default |

## Additional Context

Any other context, logs, or screenshots that may be helpful.
