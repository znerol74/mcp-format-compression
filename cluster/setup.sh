#!/bin/bash
# One-time setup on the ASC/VSC cluster
# Run this on the login node after copying files
set -e

PROJECT_DIR="$HOME/phd"
cd "$PROJECT_DIR"

echo "=== Setting up cluster environment ==="

# Load modules (adjust versions to what's available on VSC-5)
module load python/3.12.8-gcc-12.2.0-4y5tbpr

# Create single venv with vLLM + benchmark deps
echo "Creating virtual environment..."
python3 -m venv cluster-venv
source cluster-venv/bin/activate

echo "Installing vLLM..."
pip install --upgrade pip
pip install vllm

echo "Installing benchmark dependencies..."
pip install -r StableToolBench/requirements.txt
pip install -e shared_format -e toon-python -e tron-python

# Create logs directory
mkdir -p logs

# Pre-download models (login nodes usually have internet)
echo "Downloading models (this may take a while)..."
while IFS=: read -r HF_MODEL TAG NGPU QUANT PARTITION MEM; do
    [[ "$HF_MODEL" =~ ^# ]] && continue
    [ -z "$HF_MODEL" ] && continue
    echo "  Downloading $TAG ($HF_MODEL)..."
    huggingface-cli download "$HF_MODEL" || echo "  WARNING: Failed to download $HF_MODEL"
done < cluster/models.txt

# Verify StableToolBench cache
echo "Checking StableToolBench tool response cache..."
CACHE_COUNT=$(find StableToolBench/server/tool_response_cache -name "*.json" 2>/dev/null | wc -l)
echo "  Found $CACHE_COUNT cached responses"
if [ "$CACHE_COUNT" -lt 100 ]; then
    echo "  WARNING: Cache seems incomplete. Copy from local machine:"
    echo "  rsync -av local:~/phd/StableToolBench/server/tool_response_cache/ StableToolBench/server/tool_response_cache/"
fi

echo ""
echo "=== Setup complete! ==="
echo "To run benchmarks: bash cluster/run_all.sh"
echo "To run a single model: sbatch --export=... cluster/run_model.slurm"
