# ── Configuration ────────────────────────────────────────────────────
VENV  := .venv
VENV_SDFT := .venv-sdft

CUDA_HOME ?= /usr/local/cuda
export CUDA_HOME

# ── Environment Setup ────────────────────────────────────────────────
.PHONY: env env-sdft

# 1. SFT + DPO
env: $(VENV)/bin/activate

$(VENV)/bin/activate:
	@command -v uv >/dev/null 2>&1 || { echo "uv not found, installing..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	uv venv $(VENV) --python 3.11
	uv pip install --python $(VENV) -e ".[train]"
	@echo "\n✅ Environment ready. Activate with: source $(VENV)/bin/activate"

# 2. Self-distillation
env-sdft: $(VENV_SDFT)/bin/activate

$(VENV_SDFT)/bin/activate:
	@command -v uv >/dev/null 2>&1 || { echo "uv not found, installing..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	uv venv $(VENV_SDFT) --python 3.11
	uv pip install --python $(VENV_SDFT) -e ".[sdft]"
	@echo "\n✅ Environment ready. Activate with: source $(VENV_SDFT)/bin/activate"

# ── Utilities ────────────────────────────────────────────────────────
.PHONY: clean help

clean:  ## Remove all venvs
	rm -rf $(VENV) $(VENV_SDFT)

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
