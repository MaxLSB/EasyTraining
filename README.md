# EasyTraining

Config-driven DPO & SFT training with [TRL](https://github.com/huggingface/trl).

## Setup

```bash
make env
source .venv/bin/activate
```

## Training

### Single GPU

```bash
# DPO
python src/dpo/trl_dpo.py --config configs/trl_dpo_config.yaml

# SFT
python src/sft/trl_sft.py --config configs/trl_sft_config.yaml
```

### Multi-GPU (single node)

```bash
# DPO on 4 GPUs
accelerate launch --num_processes 4 src/dpo/trl_dpo.py --config configs/trl_dpo_config.yaml

# SFT on 4 GPUs
accelerate launch --num_processes 4 src/sft/trl_sft.py --config configs/trl_sft_config.yaml
```

### Multi-Node

Run on each node:

```bash
accelerate launch \
    --num_machines <NUM_NODES> \
    --num_processes <TOTAL_GPUS> \
    --machine_rank <NODE_RANK> \
    --main_process_ip <MASTER_IP> \
    --main_process_port 29500 \
    src/dpo/trl_dpo.py --config configs/trl_dpo_config.yaml
```

Or with `torchrun`:

```bash
torchrun \
    --nproc_per_node <GPUS_PER_NODE> \
    --nnodes <NUM_NODES> \
    --node_rank <NODE_RANK> \
    --master_addr <MASTER_IP> \
    --master_port 29500 \
    src/dpo/trl_dpo.py --config configs/trl_dpo_config.yaml
```
