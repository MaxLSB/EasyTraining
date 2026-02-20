#!/bin/bash
# Launch a vLLM instance with data parallelism, then run generate.py.

set -euo pipefail

# ===========================================================================
#                          EDIT YOUR CONFIG HERE
# ===========================================================================

MODEL_NAME="Qwen/Qwen3-30B-A3B-Thinking-2507"
GPUS="0,1,2,3,4,5,6,7"
PORT=8000
API_KEY="EMPTY"
TEMPERATURE=0.6
TOP_P=0.95
MAX_MODEL_LEN=18000
CONCURRENCY=32
MAX_SAMPLES=None

# ===========================================================================

IFS=',' read -ra GPU_IDS <<< "$GPUS"
NUM_GPUS=${#GPU_IDS[@]}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_SLUG="${MODEL_NAME//\//__}"
LOG_DIR="$REPO_DIR/logs"
OUTPUT_DIR="$REPO_DIR/outputs"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

BASE_URL="http://localhost:${PORT}/v1"

VLLM_PID=""
cleanup() {
    echo "Shutting down vLLM..."
    [[ -n "$VLLM_PID" ]] && kill "$VLLM_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "Launching vLLM: $MODEL_NAME (DP=$NUM_GPUS on GPUs $GPUS)"

mkdir -p /opt/home/maxence/tmp
CUDA_VISIBLE_DEVICES=$GPUS \
    TMPDIR=/opt/home/maxence/tmp \
    TORCH_HOME=/opt/home/maxence/.cache/torch \
    TORCHINDUCTOR_CACHE_DIR=/opt/home/maxence/.cache/torch/inductor \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    vllm serve "$MODEL_NAME" \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --data-parallel-size "$NUM_GPUS" \
    --trust-remote-code \
    --enable-prefix-caching \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.90 \
    > "$LOG_DIR/${MODEL_SLUG}_vllm.log" 2>&1 &
VLLM_PID=$!

# -- Wait for instance to be ready --------------------------------------------
echo "Waiting for vLLM to be ready (log: $LOG_DIR/${MODEL_SLUG}_vllm.log)..."
TIMEOUT=600
START=$(date +%s)

while true; do
    if curl -sf -H "Authorization: Bearer $API_KEY" "$BASE_URL/models" > /dev/null 2>&1; then
        echo "  vLLM is ready on port $PORT."
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM process died. Check $LOG_DIR/${MODEL_SLUG}_vllm.log"
        exit 1
    fi
    NOW=$(date +%s)
    if (( NOW - START > TIMEOUT )); then
        echo "Timeout waiting for vLLM. Exiting."
        exit 1
    fi
    sleep 5
done

# -- Run generation ------------------------------------------------------------
echo "Starting generate.py..."

MAX_SAMPLES_ARG=()
[[ "$MAX_SAMPLES" != "None" ]] && MAX_SAMPLES_ARG=(--max_samples "$MAX_SAMPLES")

python "$SCRIPT_DIR/generate.py" \
    --model_name   "$MODEL_NAME" \
    --base_urls    "$BASE_URL" \
    --api_key      "$API_KEY" \
    --concurrency  "$CONCURRENCY" \
    --temperature  "$TEMPERATURE" \
    --top_p        "$TOP_P" \
    "${MAX_SAMPLES_ARG[@]}" \
    --output_path  "$OUTPUT_DIR/${MODEL_SLUG}_generations.jsonl" \
    2>&1
