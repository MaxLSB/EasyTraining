#!/bin/bash
# Launch a single vLLM instance with data parallelism, then run loop_sanity.py.

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-0,1}"  # comma-separated GPU ids to use
IFS=',' read -ra GPU_IDS <<< "$GPUS"
NUM_GPUS=${#GPU_IDS[@]}
API_KEY="${API_KEY:-EMPTY}"
CONCURRENCY="${CONCURRENCY:-32}"
MAX_TOKENS="${MAX_TOKENS:-65536}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
NGRAM_SIZE="${NGRAM_SIZE:-20}"
NGRAM_THRESHOLD="${NGRAM_THRESHOLD:-5}"
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

BASE_URL="http://localhost:${PORT}/v1"

cleanup() {
    echo "Shutting down vLLM..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "Launching vLLM: $MODEL_NAME (DP=$NUM_GPUS on GPUs $GPUS)"

CUDA_VISIBLE_DEVICES=$GPUS vllm serve "$MODEL_NAME" \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --data-parallel-size "$NUM_GPUS" \
    --trust-remote-code \
    --enable-prefix-caching \
    > "$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

# ── Wait for instance to be ready ───────────────────────────────────────────
echo "Waiting for vLLM to be ready (log: $LOG_DIR/vllm.log)..."
TIMEOUT=600
START=$(date +%s)

while true; do
    if curl -sf -H "Authorization: Bearer $API_KEY" "$BASE_URL/models" > /dev/null 2>&1; then
        echo "  vLLM is ready on port $PORT."
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM process died. Check $LOG_DIR/vllm.log"
        exit 1
    fi
    NOW=$(date +%s)
    if (( NOW - START > TIMEOUT )); then
        echo "Timeout waiting for vLLM. Exiting."
        exit 1
    fi
    sleep 5
done

# ── Run the loop sanity check ────────────────────────────────────────────────
echo "Starting loop_sanity.py..."

python "$SCRIPT_DIR/loop_sanity.py" \
    --model_name      "$MODEL_NAME" \
    --base_urls       "$BASE_URL" \
    --api_key         "$API_KEY" \
    --concurrency     "$CONCURRENCY" \
    --max_tokens      "$MAX_TOKENS" \
    --num_samples     "$NUM_SAMPLES" \
    --seed            "$SEED" \
    --temperature     "$TEMPERATURE" \
    --top_p           "$TOP_P" \
    --ngram_size      "$NGRAM_SIZE" \
    --ngram_threshold "$NGRAM_THRESHOLD" \
    --output          "$LOG_DIR/loop_sanity_results.json"
