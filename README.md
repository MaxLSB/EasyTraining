# EasyTraining

Config-driven fine-tuning with TRL: SFT, DPO, and Self-Distillation (SDFT).

## Setup

Two separate environments due to different dependency versions:

```bash
# SFT / DPO
make env
source .venv/bin/activate

# Self-Distillation (vLLM, DeepSpeed)
make env-sdft
source .venv-sdft/bin/activate
```

```bash
hf auth login --token hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX 
wandb login
```

## Training

All methods are config-driven and support multi-GPU setups via `torchrun`. Edit the YAML then launch:

```bash
# SFT
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 src/sft/sft.py --config configs/sft_config.yaml

# DPO
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 src/dpo/dpo.py --config configs/dpo_config.yaml

# Self-Distillation
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 src/selfdistillation/sdft.py --config configs/sdft_config.yaml
```

Single GPU: drop `torchrun --nproc_per_node` and run with `python` directly.

LoRA is supported for SFT and SDFT — add a `lora:` section in the config (see examples in `configs/`).

## Cleanup

```bash
make clean  # remove all venvs
```
