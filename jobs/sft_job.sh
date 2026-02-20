#!/bin/bash -l
#SBATCH --job-name=sft                # Job name in queue
#SBATCH --partition=main              # Queue/partition to use
#SBATCH --cpus-per-task=12            # CPU cores
#SBATCH --mem=125G                    # Memory allocation
#SBATCH --gres=gpu:8                  # Request 8 GPUs
#SBATCH --time=24:00:00              # Max runtime (HH:MM:SS)
#SBATCH --output=/home/maxence_lasbordes/EasyTraining/logs/sft_%j.out   # Standard output log
#SBATCH --error=/home/maxence_lasbordes/EasyTraining/logs/sft_%j.err    # Error log

PROJECT_DIR=/home/maxence_lasbordes/EasyTraining
CONFIG="$PROJECT_DIR/configs/sft_config.yaml"
NUM_GPUS=8

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Starting at: $(date)"
echo "========================================"

mkdir -p "$PROJECT_DIR/logs"

source ~/.bashrc
source "$PROJECT_DIR/.venv/bin/activate"

# --- Launch SFT training with FSDP2 ---
echo "Starting TRL SFT training with $NUM_GPUS GPUs..."
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    "$PROJECT_DIR/src/sft/sft.py" \
    --config "$CONFIG"
EXIT_CODE=$?

echo "Finished at: $(date) with exit code $EXIT_CODE"
