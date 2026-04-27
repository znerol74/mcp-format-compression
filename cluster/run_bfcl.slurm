#!/bin/bash
#SBATCH --job-name=bfcl
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=24:00:00

# ============================================================
# Single-node BFCL job: vLLM server + 6 BFCL format-scope variants
#
# Usage:
#   sbatch --partition=zen3_0512_a100x2 --qos=zen3_0512_a100x2 --gres=gpu:1 \
#     --job-name=bfcl-mistral \
#     --export=MODEL_HF=stelterlab/Mistral-Small-24B-Instruct-2501-AWQ,MODEL_TAG=mistral-small-24b,QUANT=awq \
#     cluster/run_bfcl.slurm
#
# Optional env: NO_THINK=1, DISPLAY_NAME, ORG, BFCL_VENV (defaults to cluster-venv)
#
# Variants:
#   <TAG>            json input,  python output  (exp1 baseline)
#   <TAG>-toon       toon input,  python output  (exp1)
#   <TAG>-tron       tron input,  python output  (exp1)
#   <TAG>-json-full  json input,  json output    (exp2 baseline)
#   <TAG>-toon-full  toon input,  toon output    (exp2)
#   <TAG>-tron-full  tron input,  tron output    (exp2)
#
# Categories:
#   exp1 variants: simple_python multiple parallel parallel_multiple irrelevance
#                  multi_turn_base multi_turn_miss_func multi_turn_miss_param multi_turn_long_context
#   exp2 variants: simple_python multiple parallel parallel_multiple irrelevance  (no multi_turn — incompatible)
# ============================================================

MODEL_HF="${MODEL_HF:?must set MODEL_HF}"
MODEL_TAG="${MODEL_TAG:?must set MODEL_TAG}"
QUANT="${QUANT:-awq}"
DISPLAY_NAME="${DISPLAY_NAME:-$MODEL_TAG}"
ORG="${ORG:-Cluster}"
VLLM_PORT=8000

PROJECT_DIR="$HOME/phd"
cd "$PROJECT_DIR"
mkdir -p logs

module load python/3.12.8-gcc-12.2.0-4y5tbpr 2>/dev/null || true
source "${CLUSTER_VENV:-cluster-venv}/bin/activate"
export HF_HOME=/gpfs/data/fs73109/lkutschka/hf_cache
[ "${USE_V0}" = "1" ] && export VLLM_USE_V1=0

echo "================================================"
echo "BFCL run"
echo "Model:      $MODEL_HF ($MODEL_TAG)"
echo "GPUs:       ${SLURM_GPUS_ON_NODE:-1}"
echo "Quant:      $QUANT"
echo "Node:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "================================================"

# ---- Phase 0: Ensure BFCL model registry has our 6 entries ----
echo "[$(date '+%H:%M:%S')] Ensuring BFCL model registry entries..."
python cluster/ensure_bfcl_models.py "$MODEL_TAG" "$MODEL_HF" \
    --display "$DISPLAY_NAME" --org "$ORG"

# BFCL package must be pre-installed in the venv — install is NOT done here to avoid
# concurrent jobs racing on the shared venv (which happened in job 7838162/7838173).
# Run once manually on login node: pip install -e gorilla/berkeley-function-call-leaderboard -e shared_format -e toon-python -e tron-python
if ! python -c "import bfcl_eval" 2>/dev/null; then
    echo "ERROR: bfcl_eval not installed in $(python -c 'import sys;print(sys.prefix)')"
    echo "       Install once on login node with:"
    echo "       pip install -e gorilla/berkeley-function-call-leaderboard -e shared_format -e toon-python -e tron-python"
    exit 1
fi

# ---- Phase 1: Start vLLM server ----
echo "[$(date '+%H:%M:%S')] Starting vLLM server..."

QUANT_FLAG=""
[ "$QUANT" != "none" ] && QUANT_FLAG="--quantization $QUANT"

if [ "${NO_THINK}" = "1" ]; then
    export VLLM_ENABLE_THINKING=false
    echo "[$(date '+%H:%M:%S')] Thinking DISABLED"
fi

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_HF" \
    $QUANT_FLAG \
    --tensor-parallel-size ${SLURM_GPUS_ON_NODE:-1} \
    --port $VLLM_PORT \
    --trust-remote-code \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 8 \
    --enforce-eager \
    --dtype float16 \
    --host 0.0.0.0 \
    > logs/vllm_bfcl_${MODEL_TAG}_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

# Wait for vLLM to BOTH respond AND have the requested model loaded.
# The /v1/models endpoint can 200 before the model is registered, then /v1/chat/completions 404s.
echo "[$(date '+%H:%M:%S')] Waiting for vLLM..."
READY=0
for i in $(seq 1 180); do
    # Require the model id to actually appear in the models list
    if curl -s http://localhost:$VLLM_PORT/v1/models 2>/dev/null | grep -q "\"$MODEL_HF\""; then
        echo "[$(date '+%H:%M:%S')] vLLM ready after $((i*5))s (model registered)"
        READY=1
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM died"
        tail -50 logs/vllm_bfcl_${MODEL_TAG}_${SLURM_JOB_ID}.log
        exit 1
    fi
    sleep 5
done
[ "$READY" -ne 1 ] && { echo "ERROR: vLLM did not register $MODEL_HF in 15min"; kill $VLLM_PID 2>/dev/null; exit 1; }

# Smoke test (with retry — model may still be warming up)
SMOKE_HTTP=""
for i in 1 2 3 4 5; do
    SMOKE_HTTP=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$VLLM_PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$MODEL_HF\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hi\"}], \"max_tokens\": 10}")
    [ "$SMOKE_HTTP" = "200" ] && break
    echo "Smoke test attempt $i: HTTP $SMOKE_HTTP — retrying in 10s"
    sleep 10
done
echo "Smoke test HTTP: $SMOKE_HTTP"
[ "$SMOKE_HTTP" != "200" ] && { echo "ERROR: smoke test failed after 5 attempts"; kill $VLLM_PID 2>/dev/null; exit 1; }

# ---- Phase 2: Point BFCL handler at local vLLM ----
export OPENWEBUI_API_KEY="dummy"
export OPENWEBUI_BASE_URL="http://localhost:${VLLM_PORT}/v1"

# ---- Phase 3: Run BFCL across 6 variants ----
cd gorilla/berkeley-function-call-leaderboard

CATEGORIES_EXP1="simple_python multiple parallel parallel_multiple irrelevance multi_turn_base multi_turn_miss_func multi_turn_miss_param multi_turn_long_context"
CATEGORIES_EXP2="simple_python multiple parallel parallel_multiple irrelevance"

run_variant() {
    local key="$1"        # registry key, e.g. mistral-small-24b-toon
    local fmt="$2"        # json|toon|tron
    local tc_fmt="$3"     # python|json|toon|tron
    local categories="$4" # space-separated category names

    local override=""
    if [ "$fmt" != "json" ] || [ "$tc_fmt" != "python" ]; then
        override="ret_fmt=${tc_fmt}&tool_call_tag=False&func_doc_fmt=${fmt}&prompt_fmt=plaintext&style=classic"
    fi

    for cat in $categories; do
        echo "[$(date '+%H:%M:%S')] BFCL generate: model=$key cat=$cat fmt=$fmt tc=$tc_fmt"
        BFCL_PROMPT_FORMAT_OVERRIDE="$override" \
            python -m bfcl_eval generate \
            --model "$key" \
            --test-category "$cat" \
            --num-threads 1 \
            2>&1 | tee -a "../../logs/bfcl_${MODEL_TAG}_${key}_${cat}_${SLURM_JOB_ID}.log" \
            || echo "WARNING: $key/$cat exited with error"
    done
}

# exp1 (Python tool call output)
run_variant "${MODEL_TAG}"          json python "$CATEGORIES_EXP1"
run_variant "${MODEL_TAG}-toon"     toon python "$CATEGORIES_EXP1"
run_variant "${MODEL_TAG}-tron"     tron python "$CATEGORIES_EXP1"

# exp2 (compressed tool call output, no multi_turn)
run_variant "${MODEL_TAG}-json-full" json json "$CATEGORIES_EXP2"
run_variant "${MODEL_TAG}-toon-full" toon toon "$CATEGORIES_EXP2"
run_variant "${MODEL_TAG}-tron-full" tron tron "$CATEGORIES_EXP2"

cd "$PROJECT_DIR"

# ---- Cleanup ----
echo "[$(date '+%H:%M:%S')] Stopping vLLM..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "================================================"
echo "BFCL DONE! $(date)"
echo "Results in: gorilla/berkeley-function-call-leaderboard/result/${MODEL_TAG}*/"
echo "================================================"
