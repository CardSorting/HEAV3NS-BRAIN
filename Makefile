# HEAV3NS-BRAIN / Project SINGULARITY — Workspace Makefile
# Standard entrypoints for development, verification, and scale-to-zero operations.

SHELL := /bin/bash
PYTHON := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
RUFF := $(shell if command -v ruff >/dev/null 2>&1; then echo ruff; elif [ -x .venv/bin/ruff ]; then echo .venv/bin/ruff; else echo "$(PYTHON) -m ruff"; fi)

.PHONY: help check test test-rust test-python lint fmt fmt-check doc status audit teardown smoke-deploy

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

check: fmt-check lint doc test ## Run all formatting, linting, doc, and unit/integration test gates (offline)

test: test-rust test-python ## Run both Rust workspace and Python test suites

test-rust: ## Run all Rust unit, integration, and doc tests (offline)
	cargo test --workspace --offline

test-python: ## Run all Python unit tests in deploy/modal and scripts
	$(PYTHON) -W ignore -m unittest discover -s deploy/modal -p "test_*.py"
	$(PYTHON) -W ignore -m unittest discover -s scripts -p "test_*.py"

lint: ## Run Rust clippy (-D warnings) and Python ruff checks
	cargo clippy --all --offline -- -D warnings
	$(RUFF) check deploy/modal scripts

fmt: ## Format Rust workspace and check Python formatting
	cargo fmt --all

fmt-check: ## Verify Rust formatting without modifying files
	cargo fmt --all -- --check

doc: ## Build Rust workspace documentation with warnings denied
	RUSTDOCFLAGS="-D warnings" cargo doc --all --no-deps --offline

status: ## Show local scale-to-zero policy and provider container/app status
	$(PYTHON) -m deploy.modal.deploy status

audit: ## Verify zero-idle-compute state and detect any orphaned containers
	$(PYTHON) -m deploy.modal.deploy audit

teardown: ## Stop all known Singularity Modal apps and terminate any active containers
	$(PYTHON) -m deploy.modal.deploy teardown

smoke-deploy: ## Deploy the small-GPU (T4) smoke app for scale-to-zero lifecycle verification
	modal deploy deploy/modal/smoke_app.py

