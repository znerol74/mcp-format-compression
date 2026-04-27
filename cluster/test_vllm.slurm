#!/bin/bash
#SBATCH --job-name=test-vllm
#SBATCH --output=logs/test_vllm_%j.log
#SBATCH --time=00:10:00
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1

cd ~/phd
mkdir -p logs
module load python/3.12.8-gcc-12.2.0-4y5tbpr 2>/dev/null || true
source cluster-venv/bin/activate
export HF_HOME=/gpfs/data/fs73109/lkutschka/hf_cache

echo "=== Versions ==="
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "import tiktoken; print('tiktoken: OK')" 2>/dev/null || echo "tiktoken: MISSING"
nvidia-smi | head -5

echo ""
echo "=== Starting vLLM with Mistral-Small-24B ==="
python -m vllm.entrypoints.openai.api_server \
  --model stelterlab/Mistral-Small-24B-Instruct-2501-AWQ \
  --quantization awq \
  --port 8000 \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 8 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --host 0.0.0.0 &
VLLM_PID=$!

echo "Waiting for vLLM..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "vLLM ready after $((i*5))s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "FAILED: vLLM died"
        cat logs/test_vllm_*.log | tail -20
        exit 1
    fi
    sleep 5
done

echo ""
echo "=== Testing API call ==="
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"stelterlab/Mistral-Small-24B-Instruct-2501-AWQ","messages":[{"role":"user","content":"Say hi"}],"max_tokens":10}' | python -m json.tool

echo ""
echo "=== Testing tool calling ==="
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"stelterlab/Mistral-Small-24B-Instruct-2501-AWQ","messages":[{"role":"user","content":"What is the weather in Berlin?"}],"max_tokens":100,"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}' | python -m json.tool

echo ""
echo "=== ALL TESTS PASSED ==="
kill $VLLM_PID 2>/dev/null