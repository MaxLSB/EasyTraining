#!/bin/bash -l
#SBATCH --job-name=create_dpo       # Job name in queue
#SBATCH --partition=main            # Queue/partition to use
#SBATCH --cpus-per-task=12          # CPU cores
#SBATCH --mem=125G                  # Memory allocation
#SBATCH --gres=gpu:8                # Request 8 GPUs
#SBATCH --time=04:00:00             # Max runtime (HH:MM:SS)
#SBATCH --output=/home/maxence_lasbordes/EasyTraining/logs/create_dpo_%j.out   # Standard output log
#SBATCH --error=/home/maxence_lasbordes/EasyTraining/logs/create_dpo_%j.err    # Error log

PROJECT_DIR=/home/maxence_lasbordes/EasyTraining
MODEL_ID="Qwen/Qwen3-4B-Thinking-2507"

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

# --- Launch vLLM server with data parallelism (8 GPUs, 1 instance per GPU) ---
echo "Starting vLLM server with model $MODEL_ID (data-parallel-size=8)..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --data-parallel-size 8 \
    --max-model-len 20000 \
    --port 8000 \
    --trust-remote-code &
VLLM_PID=$!

# Wait for vLLM to be ready
echo "Waiting for vLLM server to be ready..."
for i in $(seq 1 300); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server is ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process died"
        exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start within 300s"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# --- Run dataset preparation ---
echo "Running prepare_op_dataset.py..."
python "$PROJECT_DIR/src/dpo/prepare_op_dataset.py" \
    --model_id "$MODEL_ID" \
    --max_model_len 20000

EXIT_CODE=$?

# --- Cleanup ---
echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
# Kill any remaining child processes (data-parallel workers)
kill $(jobs -p) 2>/dev/null

echo "Finished at: $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
