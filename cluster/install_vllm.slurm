#!/bin/bash
#SBATCH --job-name=install-vllm
#SBATCH --output=logs/install_vllm_%j.log
#SBATCH --time=01:00:00
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:1

cd ~/phd
mkdir -p logs
module load python/3.12.8-gcc-12.2.0-4y5tbpr
source cluster-venv/bin/activate

echo "GPU check:"
nvidia-smi | head -5
echo ""
echo "Python: $(python --version)"
echo "CUDA driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"

# Remove old vLLM completely
echo "Removing old vLLM..."
pip uninstall -y vllm 2>/dev/null

# Ensure correct PyTorch with CUDA 12.1
echo "Installing/verifying PyTorch..."
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "Installing build deps..."
pip install numpy packaging setuptools wheel

# Install vLLM with Qwen3 support
# v0.6.6 does NOT support Qwen3. Need v0.8+ which added Qwen3ForCausalLM.
# Try latest first (prebuilt wheel), fall back to specific versions
echo "Installing vLLM (need Qwen3 support)..."

# Method 1: Try latest prebuilt
pip install "vllm>=0.8" --no-build-isolation 2>&1 | tail -5

if ! python -c "import vllm; print('OK:', vllm.__version__)" 2>/dev/null; then
    echo "Method 1 failed, trying pip install from PyPI with build..."
    # Method 2: Try with build isolation (slower but more reliable)
    pip install "vllm>=0.8" 2>&1 | tail -5
fi

if ! python -c "import vllm; print('OK:', vllm.__version__)" 2>/dev/null; then
    echo "Method 2 failed, trying specific version..."
    # Method 3: Try a specific known version
    pip install vllm==0.8.5.post1 --no-build-isolation 2>&1 | tail -5
fi

if ! python -c "import vllm; print('OK:', vllm.__version__)" 2>/dev/null; then
    echo "All methods failed!"
    echo "Last resort: install from git..."
    pip install git+https://github.com/vllm-project/vllm.git 2>&1 | tail -10
fi

echo ""
echo "Verifying..."
python -c "
import vllm
print('vLLM version:', vllm.__version__)

# Check Qwen3 support by checking if the model class exists
try:
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
    print('Qwen3 support: YES')
except ImportError:
    try:
        from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
        print('Qwen3 support: likely YES (Qwen3 uses Qwen2 base in newer vLLM)')
    except:
        print('Qwen3 support: UNKNOWN')
" 2>&1

echo "DONE"