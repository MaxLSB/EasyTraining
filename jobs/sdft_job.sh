#!/bin/bash -l
#SBATCH --job-name=sdft               # Job name in queue
#SBATCH --partition=main              # Queue/partition to use
#SBATCH --cpus-per-task=12            # CPU cores
#SBATCH --mem=125G                    # Memory allocation
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --time=24:00:00              # Max runtime (HH:MM:SS)
#SBATCH --output=/home/maxence_lasbordes/EasyTraining/logs/sdft_%j.out   # Standard output log
#SBATCH --error=/home/maxence_lasbordes/EasyTraining/logs/sdft_%j.err    # Error log

PROJECT_DIR=/home/maxence_lasbordes/EasyTraining
CONFIG="$PROJECT_DIR/configs/trl_sdft_config.yaml"

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Starting at: $(date)"
echo "========================================"

mkdir -p "$PROJECT_DIR/logs"

source ~/.bashrc
source "$PROJECT_DIR/.venv-sdft/bin/activate"

export PYTHONUNBUFFERED=1

# --- Launch SDFT training (single GPU with vLLM sleep mode) ---
echo "Starting SDFT training..."
python "$PROJECT_DIR/src/selfdistillation/trl_sdft.py" \
    --config "$CONFIG"
EXIT_CODE=$?

echo "Finished at: $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
