#!/bin/bash
# Launch vLLM instances (one per GPU, data parallel) then run llm_judge.py.

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-google/gemma-3-4b-it}"
BASE_PORT="${BASE_PORT:-8000}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}"
API_KEY="${API_KEY:-EMPTY}"
INPUT_DATASET_PATH="${INPUT_DATASET_PATH:-/gpfs/projects/ehpc507/datasets/olmo_think_fr_200k/}"
OUTPUT_DATASET_PATH="${OUTPUT_DATASET_PATH:-/gpfs/projects/ehpc507/datasets/olmo_think_fr_200k_filtered/}"
NUM_WORKERS="${NUM_WORKERS:-64}"
TEMPERATURE="${TEMPERATURE:-0.15}"
TOP_P="${TOP_P:-0.7}"
# ─────────────────────────────────────────────────────────────────────────────

PIDS=()
BASE_URLS=""

cleanup() {
    echo "Shutting down vLLM instances..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "Launching $NUM_GPUS vLLM instance(s) for model: $MODEL_NAME"

for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    URL="http://localhost:${PORT}/v1"

    CUDA_VISIBLE_DEVICES=$i vllm serve "$MODEL_NAME" \
        --port "$PORT" \
        --api-key "$API_KEY" \
        --trust-remote-code \
        --enable-prefix-caching \
        &
    PIDS+=($!)

    if [ -n "$BASE_URLS" ]; then
        BASE_URLS="${BASE_URLS},${URL}"
    else
        BASE_URLS="$URL"
    fi

    echo "  GPU $i → port $PORT (PID ${PIDS[-1]})"
done

# ── Wait for all instances to be ready ───────────────────────────────────────
echo "Waiting for vLLM instances to be ready..."
TIMEOUT=600  # 10 minutes
START=$(date +%s)

for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    URL="http://localhost:${PORT}/v1/models"

    while true; do
        if curl -sf -H "Authorization: Bearer $API_KEY" "$URL" > /dev/null 2>&1; then
            echo "  Port $PORT is ready."
            break
        fi
        NOW=$(date +%s)
        if (( NOW - START > TIMEOUT )); then
            echo "Timeout waiting for vLLM on port $PORT. Exiting."
            exit 1
        fi
        sleep 5
    done
done

echo "All vLLM instances ready. Starting llm_judge.py..."

# ── Run the judge ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/llm_judge.py" \
    --input_dataset_path  "$INPUT_DATASET_PATH" \
    --output_dataset_path "$OUTPUT_DATASET_PATH" \
    --base_urls           "$BASE_URLS" \
    --api_key             "$API_KEY" \
    --model_name          "$MODEL_NAME" \
    --num_workers         "$NUM_WORKERS" \
    --temperature         "$TEMPERATURE" \
    --top_p               "$TOP_P"
