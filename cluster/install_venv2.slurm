#!/bin/bash
#SBATCH --job-name=install-v2
#SBATCH --output=logs/install_venv2_%j.log
#SBATCH --time=01:00:00
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:1

cd ~/phd
module load python/3.12.8-gcc-12.2.0-4y5tbpr 2>/dev/null || true
module load cuda/11.8.0-gcc-12.2.0-bplw5nu
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
echo "CUDA_HOME=$CUDA_HOME"
nvcc --version

echo "Creating cluster-venv2..."
rm -rf cluster-venv2
python3 -m venv cluster-venv2
source cluster-venv2/bin/activate

pip install --upgrade pip

echo "Installing build deps first..."
pip install numpy packaging setuptools wheel

echo "Installing PyTorch (CUDA 11.8) matching what vLLM prebuilt wheels expect..."
# vLLM 0.8.x cu118 wheels target torch 2.5.x; pin to avoid breakage with 2.6
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu118

echo "Installing transformers (needed before vllm builds)..."
pip install "transformers>=4.47"

echo "Installing vLLM prebuilt wheel (latest with cu118 support)..."
# Use prebuilt wheel from PyPI — no source build, no cmake.
# vLLM 0.8.5.post1 is the latest cu118 prebuild (same as cluster-venv1 but rebuilt clean).
# If we want deepseek_v3 parser we need 0.9+; try those first with --no-build-isolation fallback.
pip install "vllm==0.8.5.post1" || {
    echo "Prebuilt 0.8.5 failed, trying 0.9.x source with no-build-isolation..."
    pip install "vllm==0.9.2" --no-build-isolation
}

echo "Installing other deps..."
pip install openai requests termcolor tiktoken flask flask-cors slowapi tenacity

echo "Installing shared_format / toon / tron..."
pip install -e shared_format -e toon-python -e tron-python

echo "Verifying..."
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "
from vllm.model_executor.models import ModelRegistry
archs = ModelRegistry.get_supported_archs()
print('Qwen archs:', [m for m in archs if 'Qwen' in m])
print('Qwen3.5 MoE supported:', 'Qwen3_5MoeForConditionalGeneration' in archs)
"

echo "DONE"