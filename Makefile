VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
CUDA_HOME ?= /usr/local/cuda
export CUDA_HOME

# ── Environment ──────────────────────────────────────────────────────
.PHONY: env
env: $(VENV)/bin/activate  ## Create venv and install all dependencies

$(VENV)/bin/activate:
	@command -v uv >/dev/null 2>&1 || { echo "uv not found, installing..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	uv venv $(VENV) --python 3.11
	uv pip install -e "."
	@echo "\n✅ Environment ready. Activate with: source $(VENV)/bin/activate"

# ── Training ─────────────────────────────────────────────────────────
.PHONY: dpo sft sdft sdft-vllm

dpo:  ## Run DPO training (single GPU)
	$(PYTHON_VENV) src/dpo/trl_dpo.py --config $(CONFIG)

sft:  ## Run SFT training (single GPU)
	$(PYTHON_VENV) src/sft/trl_sft.py --config $(CONFIG)

sdft:  ## Run SDFT training (single GPU)
	$(PYTHON_VENV) src/selfdistillation/sdft.py --config $(CONFIG)

sdft-vllm:  ## Run SDFT training with vLLM generation
	$(PYTHON_VENV) src/selfdistillation/sdft_vllm.py --config $(CONFIG)

# ── Clean ────────────────────────────────────────────────────────────
.PHONY: clean
clean:  ## Remove venv
	rm -rf $(VENV)

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
