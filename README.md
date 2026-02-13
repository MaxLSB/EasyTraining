# EasyTraining

Config-driven DPO & SFT training with [TRL](https://github.com/huggingface/trl).

## Setup

```bash
make env
source .venv/bin/activate
make clean  # remove venv
```

```bash
hf auth login --token hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX # Set you HF token
wandb login # Set your wandb key
```



## Training

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 src/dpo/trl_dpo.py --config configs/trl_dpo_config.yaml > logs/dpo_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 src/sft/trl_sft.py --config configs/trl_sft_config.yaml > logs/sft_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 src/selfdistillation/sdft.py --config configs/sdft_config.yaml > logs/sdft_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 src/selfdistillation/sdft_vllm.py --config configs/sdft_vllm_config.yaml > logs/sdft_vllm_test.log 2>&1 &
```