#!/bin/bash
#SBATCH --job-name=fmt-bench
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=24:00:00

# ============================================================
# Single-node job: starts vLLM server + runs StableToolBench
# Both run on the same GPU node, communicate via localhost
#
# Usage:
#   sbatch --partition=zen3_0512_a100x2 --qos=zen3_0512_a100x2 --gres=gpu:1 --time=24:00:00 \
#     --export=MODEL_HF=Qwen/Qwen3-32B-AWQ,MODEL_TAG=qwen3-32b,QUANT=awq \
#     cluster/run_model.slurm
# ============================================================

MODEL_HF="${MODEL_HF:-Qwen/Qwen3-32B-AWQ}"
MODEL_TAG="${MODEL_TAG:-qwen3-32b}"
QUANT="${QUANT:-awq}"
TOOL_PARSER="${TOOL_PARSER:-hermes}"
TOKENIZER_MODE="${TOKENIZER_MODE:-auto}"
TOKENIZER_NAME="${TOKENIZER_NAME:-}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
VLLM_PORT=8000
STB_PORT=8080
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

PROJECT_DIR="$HOME/phd"
cd "$PROJECT_DIR"
mkdir -p logs

# Load modules
module load python/3.12.8-gcc-12.2.0-4y5tbpr 2>/dev/null || true
source "${CLUSTER_VENV:-cluster-venv}/bin/activate"
export HF_HOME=/gpfs/data/fs73109/lkutschka/hf_cache
# Use V0 engine if requested (needed for some MoE models not yet in V1)
[ "${USE_V0}" = "1" ] && export VLLM_USE_V1=0

echo "================================================"
echo "Model:      $MODEL_HF ($MODEL_TAG)"
echo "GPUs:       ${SLURM_GPUS_ON_NODE:-1}"
echo "Quant:      $QUANT"
echo "Tool parser: $TOOL_PARSER"
echo "Timestamp:  $TIMESTAMP"
echo "Node:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "Python:     $(which python)"
echo "vLLM:       $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
echo "================================================"

# Verify vLLM is installed and supports the model
python -c "import vllm; print('vLLM OK:', vllm.__version__)" || { echo "ERROR: vLLM not installed"; exit 1; }

# ---- Phase 1: Start vLLM server in background ----
echo "[$(date '+%H:%M:%S')] Starting vLLM server..."

QUANT_FLAG=""
[ "$QUANT" != "none" ] && QUANT_FLAG="--quantization $QUANT"

# Disable thinking if NO_THINK is set
THINK_FLAG=""
if [ "${NO_THINK}" = "1" ]; then
    # vLLM 0.8.5 doesn't support --chat-template-kwargs
    # Instead, set via environment variable that the chat template reads
    export VLLM_ENABLE_THINKING=false
    echo "[$(date '+%H:%M:%S')] Thinking DISABLED (via env)"
else
    echo "[$(date '+%H:%M:%S')] Thinking ENABLED (default)"
fi

# Force float16 for quantized models (GPTQ/AWQ may not support bfloat16)
DTYPE_FLAG="--dtype float16"

# Optional flags for models needing special tokenizer/template (e.g. Mistral)
TOKENIZER_FLAGS=""
[ -n "$TOKENIZER_NAME" ] && TOKENIZER_FLAGS="$TOKENIZER_FLAGS --tokenizer $TOKENIZER_NAME"
[ "$TOKENIZER_MODE" != "auto" ] && TOKENIZER_FLAGS="$TOKENIZER_FLAGS --tokenizer-mode $TOKENIZER_MODE"
[ -n "$CHAT_TEMPLATE" ] && TOKENIZER_FLAGS="$TOKENIZER_FLAGS --chat-template $CHAT_TEMPLATE"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_HF" \
    $QUANT_FLAG \
    --tensor-parallel-size ${SLURM_GPUS_ON_NODE:-1} \
    --port $VLLM_PORT \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 8 \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser $TOOL_PARSER \
    $TOKENIZER_FLAGS \
    $DTYPE_FLAG \
    --host 0.0.0.0 \
    > logs/vllm_${MODEL_TAG}_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

# Wait for vLLM AND for the requested model to actually be registered.
# /v1/models can return 200 for a lingering server from another job — we must verify
# that OUR model appears in the response, not just that some server answers.
echo "[$(date '+%H:%M:%S')] Waiting for vLLM to register $MODEL_HF..."
READY=0
for i in $(seq 1 180); do
    if curl -s http://localhost:$VLLM_PORT/v1/models 2>/dev/null | grep -q "\"$MODEL_HF\""; then
        echo "[$(date '+%H:%M:%S')] vLLM ready after $((i*5))s (model registered)"
        READY=1
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died! Check logs/vllm_${MODEL_TAG}_${SLURM_JOB_ID}.log"
        tail -50 logs/vllm_${MODEL_TAG}_${SLURM_JOB_ID}.log
        exit 1
    fi
    sleep 5
done

if [ "$READY" -ne 1 ]; then
    echo "ERROR: vLLM did not register $MODEL_HF within 15 minutes"
    tail -50 logs/vllm_${MODEL_TAG}_${SLURM_JOB_ID}.log
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Show loaded model info
echo "Loaded model:"
curl -s http://localhost:$VLLM_PORT/v1/models | python -m json.tool

# Quick smoke test (with retry for warm-up)
echo "[$(date '+%H:%M:%S')] Smoke test..."
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
if [ "$SMOKE_HTTP" != "200" ]; then
    echo "ERROR: Model smoke test failed after retries (HTTP $SMOKE_HTTP)"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi
echo "Smoke test PASSED"

# ---- Phase 2: Set environment ----
# vLLM serves the model under its HuggingFace name
export OPENWEBUI_API_KEY="dummy"
export OPENWEBUI_BASE_URL="http://localhost:${VLLM_PORT}/v1"

# ---- Phase 3: Run StableToolBench ----
echo "[$(date '+%H:%M:%S')] Running StableToolBench..."
cd StableToolBench
export PYTHONPATH=./:./toolbench/inference
export SERVICE_URL="http://localhost:${STB_PORT}/virtual"

# Start virtual API server (tool response cache)
echo "[$(date '+%H:%M:%S')] Starting virtual API server on port ${STB_PORT}..."
(cd server && python main.py) > ../logs/toolserver_${SLURM_JOB_ID}.log 2>&1 &
TOOL_PID=$!
sleep 10

# Verify tool server
TOOL_CHECK=$(curl -s http://localhost:${STB_PORT}/virtual -X POST \
    -H "Content-Type: application/json" \
    -d '{"category":"Data","tool_name":"test","api_name":"test","tool_input":"{}","strip":"truncate","toolbench_key":""}' 2>&1)
if echo "$TOOL_CHECK" | grep -q "error"; then
    echo "Virtual API server responding (may have errors for test query, this is OK)"
else
    echo "WARNING: Virtual API server may not be responding"
    echo "Response: $TOOL_CHECK"
fi

# Run exp2 (full compression) for all 3 formats
# The model name sent to vLLM must match the HuggingFace model name
for fmt in json toon tron; do
    for group in G1 G2 G3; do
        OUTDIR="results/${MODEL_TAG}_${fmt}_tc${fmt}/${group}_instruction"
        echo "[$(date '+%H:%M:%S')] Starting ${group}/${fmt}..."
        python toolbench/inference/qa_pipeline.py \
            --tool_root_dir server/tools \
            --openwebui_model "$MODEL_HF" \
            --method CoT@1 \
            --input_query_file "solvable_queries/test_instruction/${group}_instruction.json" \
            --output_answer_file "$OUTDIR" \
            --tool_format "$fmt" \
            --tool_call_format "$fmt" \
            2>&1 | tee -a ../logs/stb_${MODEL_TAG}_${fmt}_${SLURM_JOB_ID}.log \
            || echo "WARNING: ${group}/${fmt} exited with error"
        echo "[$(date '+%H:%M:%S')] Done ${group}/${fmt}"
    done
done

kill $TOOL_PID 2>/dev/null
cd "$PROJECT_DIR"

# ---- Cleanup ----
echo "[$(date '+%H:%M:%S')] Stopping vLLM..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "================================================"
echo "ALL DONE! $(date)"
echo "Results in: StableToolBench/results/${MODEL_TAG}_*/"
echo "================================================"