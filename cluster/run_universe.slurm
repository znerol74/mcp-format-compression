#!/bin/bash
#SBATCH --job-name=universe
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=48:00:00

# ============================================================
# Single-node MCP-Universe job: vLLM server + 5 format-scope variants
#
# Usage:
#   sbatch --partition=zen3_0512_a100x2 --qos=zen3_0512_a100x2 --gres=gpu:1 \
#     --job-name=universe-mistral \
#     --export=MODEL_HF=stelterlab/Mistral-Small-24B-Instruct-2501-AWQ,MODEL_TAG=mistral-small-24b,QUANT=awq \
#     cluster/run_universe.slurm
#
# Optional env: NO_THINK=1, CATEGORIES (override default list), SERP_API_KEY, GOOGLE_MAPS_API_KEY, GITHUB_PERSONAL_ACCESS_TOKEN
#
# Categories (per-server requirements):
#   financial_analysis        — yfinance+calculator (pure Python, no key)            ✓ easy
#   location_navigation       — Google Maps (needs GOOGLE_MAPS_API_KEY)              ✓ easy
#   web_search                — SerpAPI (needs SERP_API_KEY)                         ⚠ rate-limited
#   browser_automation        — Playwright via npx (needs Node.js + browser binary)  ⚠ medium
#   3d_design                 — Blender Python module                                ⚠ medium
#   repository_management     — GitHub MCP via Docker                                ✗ no Docker on compute
#
# Default CATEGORIES skips repository_management. Override via --export=CATEGORIES="..."
# ============================================================

MODEL_HF="${MODEL_HF:?must set MODEL_HF}"
MODEL_TAG="${MODEL_TAG:?must set MODEL_TAG}"
QUANT="${QUANT:-awq}"
CATEGORIES="${CATEGORIES:-financial_analysis web_search}"
VLLM_PORT=8000

PROJECT_DIR="$HOME/phd"
cd "$PROJECT_DIR"
mkdir -p logs

module load python/3.12.8-gcc-12.2.0-4y5tbpr 2>/dev/null || true
source "${CLUSTER_VENV:-cluster-venv}/bin/activate"
export HF_HOME=/gpfs/data/fs73109/lkutschka/hf_cache
[ "${USE_V0}" = "1" ] && export VLLM_USE_V1=0

echo "================================================"
echo "MCP-Universe run"
echo "Model:      $MODEL_HF ($MODEL_TAG)"
echo "Categories: $CATEGORIES"
echo "GPUs:       ${SLURM_GPUS_ON_NODE:-1}"
echo "Node:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "================================================"

# ---- Phase 0: Install Universe deps if not yet ----
if ! python -c "import mcpuniverse" 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] Installing MCP-Universe..."
    cd MCP-Universe
    pip install -e .
    pip install -e ../shared_format -e ../toon-python -e ../tron-python
    cd ..
fi

# Source MCP-Universe .env if present (provides tool API keys: SERP, GOOGLE_MAPS, etc.)
if [ -f MCP-Universe/.env ]; then
    set -a; source MCP-Universe/.env; set +a
fi

# Override any .env values that would route LLM calls away from localhost vLLM.
# MCP-Universe's OpenAIModel reads OPENAI_BASE_URL / OPENAI_API_KEY as pydantic defaults;
# clearing these after sourcing .env ensures our yaml-mutated base_url = localhost:$VLLM_PORT wins.
export OPENAI_API_KEY="dummy"
export OPENAI_BASE_URL="http://localhost:${VLLM_PORT}/v1"

# Verify required API keys for each category
need_key() {
    local var="$1"; local cat="$2"
    if [ -z "${!var}" ]; then
        echo "WARNING: $var not set — $cat tasks will likely fail"
    fi
}
echo "$CATEGORIES" | grep -qw web_search          && need_key SERP_API_KEY web_search
echo "$CATEGORIES" | grep -qw location_navigation && need_key GOOGLE_MAPS_API_KEY location_navigation
echo "$CATEGORIES" | grep -qw repository_management && need_key GITHUB_PERSONAL_ACCESS_TOKEN repository_management

# ---- Phase 1: Start vLLM server ----
echo "[$(date '+%H:%M:%S')] Starting vLLM server..."

QUANT_FLAG=""
[ "$QUANT" != "none" ] && QUANT_FLAG="--quantization $QUANT"

if [ "${NO_THINK}" = "1" ]; then
    export VLLM_ENABLE_THINKING=false
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
    > logs/vllm_universe_${MODEL_TAG}_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

# Wait for vLLM AND model registration (not just /v1/models responding)
for i in $(seq 1 180); do
    if curl -s http://localhost:$VLLM_PORT/v1/models 2>/dev/null | grep -q "\"$MODEL_HF\""; then
        echo "[$(date '+%H:%M:%S')] vLLM ready after $((i*5))s (model registered)"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM died"; tail -50 logs/vllm_universe_${MODEL_TAG}_${SLURM_JOB_ID}.log; exit 1
    fi
    sleep 5
done

# ---- Phase 2: Generate per-(category, fmt, scope) YAML configs ----
# YAML mutation: set llm.spec.type=openai with localhost base_url, set agent.spec.config.tool_call_format
TMP_CFG_DIR="logs/universe_configs/${MODEL_TAG}_${SLURM_JOB_ID}"
mkdir -p "$TMP_CFG_DIR"

python <<PYEOF
import os, sys, yaml
from pathlib import Path

base_dir = Path("MCP-Universe/mcpuniverse/benchmark/configs/mcpuniverse/test_full")
out_dir = Path("$TMP_CFG_DIR")
model_hf = "$MODEL_HF"
categories = "$CATEGORIES".split()

# (fmt, tc_fmt) variants. (json, json) covers both exp1 and exp2 baseline.
variants = [("json", "json"), ("toon", "json"), ("tron", "json"),
            ("toon", "toon"), ("tron", "tron")]

for cat in categories:
    for fmt, tc_fmt in variants:
        base_path = base_dir / (f"{cat}.yaml" if fmt == "json" else f"{cat}_{fmt}.yaml")
        if not base_path.exists():
            print(f"  SKIP (no config): {base_path}", file=sys.stderr); continue
        with open(base_path) as f:
            docs = list(yaml.safe_load_all(f))
        for doc in docs:
            if doc.get("kind") == "llm":
                doc["spec"]["type"] = "openai"
                doc["spec"]["config"]["model_name"] = model_hf
                doc["spec"]["config"]["base_url"] = "http://localhost:${VLLM_PORT}/v1"
                doc["spec"]["config"]["api_key"] = "dummy"
                doc["spec"]["config"]["verify_ssl"] = False
            if doc.get("kind") == "agent":
                doc["spec"]["config"]["tool_call_format"] = tc_fmt
        out_path = out_dir / f"{cat}_{fmt}_tc{tc_fmt}.yaml"
        with open(out_path, "w") as f:
            yaml.dump_all(docs, f, default_flow_style=False)
        print(f"  wrote {out_path}")
PYEOF

# ---- Phase 3: Run Universe across all (category, variant) configs ----
cd MCP-Universe

# MCP servers launched as subprocesses resolve "python3" via shutil.which() — venv must be on PATH
export PATH="$PROJECT_DIR/${CLUSTER_VENV:-cluster-venv}/bin:$PATH"

for config in "$PROJECT_DIR/$TMP_CFG_DIR"/*.yaml; do
    name=$(basename "$config" .yaml)
    log="$PROJECT_DIR/logs/universe_${MODEL_TAG}_${name}_${SLURM_JOB_ID}.log"
    echo "[$(date '+%H:%M:%S')] Universe: $name"
    python run_full_test.py "$config" 2>&1 | tee -a "$log" \
        || echo "WARNING: $name exited with error"
done

cd "$PROJECT_DIR"

# ---- Cleanup ----
echo "[$(date '+%H:%M:%S')] Stopping vLLM..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "================================================"
echo "Universe DONE! $(date)"
echo "Reports in: MCP-Universe/test_full_report/"
echo "================================================"
