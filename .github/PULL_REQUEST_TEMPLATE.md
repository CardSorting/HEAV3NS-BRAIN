## Summary

<!-- A concise description of what this PR does and why. -->

## Type of Change

- [ ] `feat` — New feature
- [ ] `fix` — Bug fix
- [ ] `perf` — Performance improvement
- [ ] `refactor` — Code refactor (no behavior change)
- [ ] `docs` — Documentation only
- [ ] `test` — Test additions or changes
- [ ] `chore` — Build, CI, dependency updates

## Affected Crate(s)

- [ ] `singularity-core`
- [ ] `singularity-cuda`
- [ ] `singularity-cache`
- [ ] `singularity-scheduler`
- [ ] `singularity-model`
- [ ] `singularity-grammar`
- [ ] `singularity-server`
- [ ] `singularity-cli`
- [ ] `singularity-tests`
- [ ] Infrastructure

## Hardware Path

- [ ] SM100 (Blackwell B200) — FP4 path
- [ ] SM90 (Hopper H200) — MXFP8 path
- [ ] Both
- [ ] CPU mock / hardware-agnostic

## Checklist

- [ ] `cargo test --all` passes (all 61+ tests green)
- [ ] `cargo clippy --all -- -D warnings` passes with no errors
- [ ] `cargo fmt --all` applied
- [ ] New public items have `///` doc comments
- [ ] Any `unsafe` blocks include a `// SAFETY:` justification comment
- [ ] New features include at least one test
- [ ] Performance-sensitive changes include a Criterion benchmark
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

## Related Issues

Closes #<!-- issue number -->

## Testing Evidence

<!-- Paste relevant test output or benchmark comparison here. -->

```
cargo test --all
...
test result: ok. XX passed; 0 failed
```

