#!/bin/bash
#SBATCH --job-name=benchpp
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=48:00:00

# ============================================================
# Single-node BenchPP job: 2x vLLM (model + judge) on 2 GPUs + 6 BenchPP variants
#
# Usage:
#   sbatch --partition=zen3_0512_a100x2 --qos=zen3_0512_a100x2 --gres=gpu:2 \
#     --job-name=benchpp-mistral \
#     --export=MODEL_HF=stelterlab/Mistral-Small-24B-Instruct-2501-AWQ,MODEL_TAG=mistral-small-24b,QUANT=awq \
#     cluster/run_benchpp.slurm
#
# Optional env: NO_THINK=1, JUDGE_HF, JUDGE_QUANT
# Default judge: Qwen3-32B-AWQ (smaller than DeepSeek-V3 used locally; runs on cluster)
#
# How it dispatches model vs judge calls:
#   Model under test: name passed with "-openwebui" suffix → BenchPP routes via OPENWEBUI_URL=localhost:8000
#   Judge:            plain HF name → falls through to CUSTOM_OPENAI_BASE_URL=localhost:8001
#   (We deliberately do NOT set OPENROUTER_API_KEY so dispatch hits the CustomOpenAI branch.)
# ============================================================

MODEL_HF="${MODEL_HF:?must set MODEL_HF}"
MODEL_TAG="${MODEL_TAG:?must set MODEL_TAG}"
QUANT="${QUANT:-awq}"
JUDGE_HF="${JUDGE_HF:-Qwen/Qwen3-32B-AWQ}"
JUDGE_QUANT="${JUDGE_QUANT:-awq}"
# TP size for model under test. FP16 32B doesn't fit on one A100 40GB — set MODEL_TP=2 and
# request --gres=gpu:3 so model can use GPUs 0+1 and judge uses GPU 2.
MODEL_TP="${MODEL_TP:-1}"
MODEL_PORT=8000
JUDGE_PORT=8001

PROJECT_DIR="$HOME/phd"
cd "$PROJECT_DIR"
mkdir -p logs

module load python/3.12.8-gcc-12.2.0-4y5tbpr 2>/dev/null || true
source "${CLUSTER_VENV:-cluster-venv}/bin/activate"
export HF_HOME=/gpfs/data/fs73109/lkutschka/hf_cache
[ "${USE_V0}" = "1" ] && export VLLM_USE_V1=0

echo "================================================"
echo "MCPToolBenchPP run"
echo "Model under test: $MODEL_HF ($MODEL_TAG)"
echo "Judge:            $JUDGE_HF"
echo "GPUs:             ${SLURM_GPUS_ON_NODE:-2}"
echo "Node:             $(hostname)"
echo "Job ID:           $SLURM_JOB_ID"
echo "================================================"

MIN_GPUS=$((MODEL_TP + 1))
if [ "${SLURM_GPUS_ON_NODE:-0}" -lt "$MIN_GPUS" ]; then
    echo "ERROR: this job needs --gres=gpu:$MIN_GPUS (MODEL_TP=$MODEL_TP model GPUs + 1 judge GPU)"
    exit 1
fi

# Install BenchPP deps if not yet
if ! python -c "import openai, requests, yaml" 2>/dev/null; then
    pip install openai requests pyyaml
fi
# Ensure shared_format etc. are installed
if ! python -c "import shared_format" 2>/dev/null; then
    pip install -e shared_format -e toon-python -e tron-python
fi

# ---- Phase 1: Start MODEL vLLM on GPU 0 (port 8000) ----
echo "[$(date '+%H:%M:%S')] Starting MODEL vLLM on GPU 0..."

QUANT_FLAG=""
[ "$QUANT" != "none" ] && QUANT_FLAG="--quantization $QUANT"

if [ "${NO_THINK}" = "1" ]; then
    export VLLM_ENABLE_THINKING=false
fi

# Model uses GPUs 0..MODEL_TP-1; judge uses MODEL_TP
MODEL_GPUS=$(seq -s, 0 $((MODEL_TP - 1)))
JUDGE_GPU=$MODEL_TP

CUDA_VISIBLE_DEVICES=$MODEL_GPUS python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_HF" \
    $QUANT_FLAG \
    --tensor-parallel-size $MODEL_TP \
    --port $MODEL_PORT \
    --trust-remote-code \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 8 \
    --enforce-eager \
    --dtype float16 \
    --host 0.0.0.0 \
    > logs/vllm_benchpp_model_${MODEL_TAG}_${SLURM_JOB_ID}.log 2>&1 &
MODEL_PID=$!

# ---- Phase 1b: Start JUDGE vLLM on GPU 1 (port 8001) ----
echo "[$(date '+%H:%M:%S')] Starting JUDGE vLLM on GPU 1..."

JUDGE_QUANT_FLAG=""
[ "$JUDGE_QUANT" != "none" ] && JUDGE_QUANT_FLAG="--quantization $JUDGE_QUANT"

CUDA_VISIBLE_DEVICES=$JUDGE_GPU python -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_HF" \
    $JUDGE_QUANT_FLAG \
    --tensor-parallel-size 1 \
    --port $JUDGE_PORT \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 4 \
    --enforce-eager \
    --dtype float16 \
    --host 0.0.0.0 \
    > logs/vllm_benchpp_judge_${MODEL_TAG}_${SLURM_JOB_ID}.log 2>&1 &
JUDGE_PID=$!

# Wait for both
wait_for_port() {
    local port=$1; local pid=$2; local label=$3; local model=$4
    for i in $(seq 1 180); do
        if curl -s http://localhost:$port/v1/models 2>/dev/null | grep -q "\"$model\""; then
            echo "[$(date '+%H:%M:%S')] $label ready after $((i*5))s (model registered)"
            return 0
        fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "ERROR: $label process died"
            return 1
        fi
        sleep 5
    done
    echo "ERROR: $label did not register model in 15min"; return 1
}
wait_for_port $MODEL_PORT $MODEL_PID "MODEL" "$MODEL_HF"  || { kill $MODEL_PID $JUDGE_PID 2>/dev/null; exit 1; }
wait_for_port $JUDGE_PORT $JUDGE_PID "JUDGE" "$JUDGE_HF"  || { kill $MODEL_PID $JUDGE_PID 2>/dev/null; exit 1; }

# ---- Phase 2: Configure BenchPP env vars ----
# Model under test: routed via OPENWEBUI handler (any model name ending in -openwebui)
export OPENWEBUI_URL="http://localhost:${MODEL_PORT}/v1"
export OPENWEBUI_API_KEY="dummy"

# Judge: routed via CustomOpenAI handler (catch-all for non-OpenRouter, non-OpenWebUI)
export CUSTOM_OPENAI_BASE_URL="http://localhost:${JUDGE_PORT}/v1"
export CUSTOM_OPENAI_API_KEY="dummy"

# Critical: do NOT set OPENROUTER_API_KEY (would override our judge dispatch)
unset OPENROUTER_API_KEY

# ---- Phase 3: Run BenchPP across 6 variants × N categories ----
cd MCPToolBenchPP

# Categories with their input data files
declare -a CATEGORIES=(
    "finance:./data/finance/finance_0724_single_v3.json"
    "pay:./data/pay/pay_0723_single.json"
    "search:./data/search/search_0723_single.json"
    "file_system:./data/file_system/filesystem_0723_single.json"
    "map:./data/map/map_0717_single_multi_lang_500.json"
    "browser:./data/browser/browser_0724_single_v3.json"
)

# Format scopes to run: (fmt, tc_fmt) pairs
# exp1: tool_call_format=json (or python — but BenchPP uses json native)
# exp2: tool_call_format=fmt
declare -a VARIANTS=(
    "json:json"   # exp1 + exp2 baseline (both same when fmt=json)
    "toon:json"   # exp1 TOON
    "tron:json"   # exp1 TRON
    "toon:toon"   # exp2 TOON
    "tron:tron"   # exp2 TRON
)
# (json:json appears once — it covers both exp1 and exp2 baseline since they're identical)

MODEL_OPENWEBUI="${MODEL_HF}-openwebui"

for variant in "${VARIANTS[@]}"; do
    fmt="${variant%:*}"
    tc_fmt="${variant#*:}"
    for catpair in "${CATEGORIES[@]}"; do
        cat="${catpair%:*}"
        input="${catpair#*:}"
        log="../logs/benchpp_${MODEL_TAG}_${fmt}_tc${tc_fmt}_${cat}_${SLURM_JOB_ID}.log"
        # Deterministic log_file so resubmits RESUME instead of starting fresh.
        # (qa_pipeline skips already-completed tasks when --log_file points at an existing file.)
        input_stem=$(basename "$input" .json)
        resume_log="${input_stem}_${fmt}_tc${tc_fmt}_${MODEL_TAG}_resume.json"
        echo "[$(date '+%H:%M:%S')] BenchPP: model=$MODEL_TAG cat=$cat fmt=$fmt tc=$tc_fmt resume=$resume_log"
        python run.py \
            --stage tool_call \
            --input_file "$input" \
            --category "$cat" \
            --model "$MODEL_OPENWEBUI" \
            --tool_mode prompt \
            --tool_format "$fmt" \
            --tool_call_format "$tc_fmt" \
            --llm_as_judge_model "$JUDGE_HF" \
            --log_file "$resume_log" \
            --pass_k 1 \
            --evaluation_trial_per_task 5 \
            2>&1 | tee -a "$log" \
            || echo "WARNING: $cat/$fmt exited with error"
    done
done

cd "$PROJECT_DIR"

# ---- Cleanup ----
echo "[$(date '+%H:%M:%S')] Stopping vLLM servers..."
kill $MODEL_PID $JUDGE_PID 2>/dev/null
wait $MODEL_PID $JUDGE_PID 2>/dev/null

echo "================================================"
echo "BenchPP DONE! $(date)"
echo "Results in: MCPToolBenchPP/logs/"
echo "================================================"
