#!/bin/bash
#SBATCH --job-name=eval-judge
#SBATCH --output=logs/eval_judge_%j.log
#SBATCH --time=3-00:00:00
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:1

# ============================================================
# Run StableToolBench evaluation using a local LLM judge
# Uses DeepSeek-R1-Distill-32B as judge (for non-DeepSeek results)
# Uses Qwen3-32B-AWQ as judge (for DeepSeek results)
# ============================================================

cd ~/phd
mkdir -p logs
module load python/3.12.8-gcc-12.2.0-4y5tbpr 2>/dev/null || true
source cluster-venv/bin/activate
export HF_HOME=/gpfs/data/fs73109/lkutschka/hf_cache

JUDGE_MODEL="${JUDGE_MODEL:-casperhansen/deepseek-r1-distill-qwen-32b-awq}"
JUDGE_TAG="${JUDGE_TAG:-deepseek-r1-32b}"
VLLM_PORT=8000

echo "================================================"
echo "Evaluation Judge Server"
echo "Judge:      $JUDGE_MODEL ($JUDGE_TAG)"
echo "Node:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "================================================"

# ---- Start vLLM judge server ----
echo "[$(date '+%H:%M:%S')] Starting judge server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" \
    --quantization awq \
    --port $VLLM_PORT \
    --trust-remote-code \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 16 \
    --enforce-eager \
    --dtype float16 \
    --host 0.0.0.0 \
    > logs/vllm_judge_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

# Wait for ready
echo "[$(date '+%H:%M:%S')] Waiting for judge to load..."
READY=0
for i in $(seq 1 120); do
    if curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null 2>&1; then
        echo "[$(date '+%H:%M:%S')] Judge ready after $((i*5))s"
        READY=1
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: Judge server died!"
        tail -20 logs/vllm_judge_${SLURM_JOB_ID}.log
        exit 1
    fi
    sleep 5
done

if [ "$READY" -ne 1 ]; then
    echo "ERROR: Judge did not start"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Smoke test
SMOKE_HTTP=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$VLLM_PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$JUDGE_MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Is this task solved? Answer: Solved or Unsolved.\"}], \"max_tokens\": 10}")
echo "Judge smoke test HTTP: $SMOKE_HTTP"

# ---- Create eval API pool pointing to local vLLM ----
cat > /tmp/eval_api_pool_local.json << EOF
[
    {
        "api_key": "dummy",
        "api_base": "http://localhost:${VLLM_PORT}/v1",
        "model": "${JUDGE_MODEL}"
    }
]
EOF

# ---- Create evaluator config for local judge ----
EVAL_DIR="StableToolBench/toolbench/tooleval/evaluators/tooleval_local_judge"
mkdir -p "$EVAL_DIR"

# Copy template from existing evaluator
cp StableToolBench/toolbench/tooleval/evaluators/tooleval_gpt4o_mini/template.txt "$EVAL_DIR/" 2>/dev/null || \
    cp StableToolBench/toolbench/tooleval/evaluators/tooleval_gpt-3.5-turbo_default/template.txt "$EVAL_DIR/"

cat > "$EVAL_DIR/config.yaml" << EOF
evaluator_name: "tooleval_local_judge"
registered_cls_name: "ReinforceToolLearningEvaluator"
prompt_template: "template.txt"
fn_completions: "normalized_openai_completions"
apis_json: "/tmp/eval_api_pool_local.json"
completions_kwargs:
  model: "${JUDGE_MODEL}"
  max_tokens: 1000
  temperature: 0.2
  timeout: 30
  functions:
    - name: "check_answer_status"
      description: "Parse the json answer with layerd nodes and return the answer_status about the answer"
      parameters:
        type: "object"
        properties:
          answer_status:
            type: "string"
            enum: ["Unsure","Unsolved","Solved"]
        required: ["answer_status"]
    - name: "parse_answer_status"
      description: "Parse the json answer with layerd nodes and return the answer_status about the answer"
      parameters:
        type: "object"
        properties:
          answer_status:
            type: "string"
            enum: ["Unsure","Unsolved","Solved"]
        required: ["answer_status"]
    - name: "check_task_solvable"
      description: "Parse the task description and return the task_status about the task"
      parameters:
        type: "object"
        properties:
          task_status:
            type: "string"
            enum: ["Unsure","Unsolvable","Solvable"]
        required: ["task_status"]
    - name: "select_better_answer"
      description: "Select the better answer"
      parameters:
        type: "object"
        properties:
          index:
            type: "number"
            description: "The index value in the selected better answer."
        required: ["index"]
fn_completion_parser: "index_parser"
batch_size: 1
EOF

# ---- Run evaluations ----
cd StableToolBench
export PYTHONPATH=./:./toolbench/inference:./toolbench/tooleval
export API_POOL_FILE="/tmp/eval_api_pool_local.json"

# List all model result dirs to evaluate
MODELS=(
    "local_json_tcjson"
    "local_toon_tctoon"
    "local_tron_tctron"
    "qwen35_json_tcjson"
    "qwen35_toon_tctoon"
    "qwen35_tron_tctron"
    "qwen3-32b-fp16_json_tcjson"
    "qwen3-32b-fp16_toon_tctoon"
    "qwen3-32b-fp16_tron_tctron"
    "qwen3-32b-awq_json_tcjson"
    "qwen3-32b-awq_toon_tctoon"
    "qwen3-32b-awq_tron_tctron"
    "mistral-small-24b_json_tcjson"
    "mistral-small-24b_toon_tctoon"
    "mistral-small-24b_tron_tctron"
    "deepseek-r1-32b_json_tcjson"
    "deepseek-r1-32b_toon_tctoon"
    "deepseek-r1-32b_tron_tctron"
)

for tag in "${MODELS[@]}"; do
    src="results/${tag}"
    if [ ! -d "$src" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP $tag (no results dir)"
        continue
    fi

    echo "[$(date '+%H:%M:%S')] Evaluating $tag..."

    # Convert results
    dst="eval_converted_local/${tag}"
    mkdir -p "$dst"

    python -c "
import json, os
src = '$src'
dst = '$dst'
for group in ['G1_instruction', 'G2_instruction', 'G3_instruction']:
    gd = os.path.join(src, group)
    if not os.path.exists(gd): continue
    converted = {}
    for f in os.listdir(gd):
        if not f.endswith('.json'): continue
        qid = f.replace('_CoT@1.json','')
        data = json.load(open(os.path.join(gd, f)))
        ag = data.get('answer_generation', {})
        converted[qid] = {
            'query': ag.get('query', ''),
            'available_tools': [f2 for f2 in ag.get('function', [])] if ag.get('function') else [],
            'answer': {'method': 'CoT@1', 'total_steps': 0, 'final_answer': ag.get('final_answer', ''), 'answer_details': [{'role': 'assistant', 'message': ag.get('final_answer', ''), 'next': []}]}
        }
    json.dump(converted, open(os.path.join(dst, f'{group}.json'), 'w'), ensure_ascii=False, indent=2)
    print(f'  Converted {group}: {len(converted)} tasks')
"

    # Run evaluation
    python -c "
import json, os, sys, random
sys.path.insert(0, 'toolbench/tooleval')
sys.path.insert(0, 'toolbench/inference')
os.environ['API_POOL_FILE'] = '/tmp/eval_api_pool_local.json'
from evaluators import load_registered_automatic_evaluator
from evaluators.registered_cls.rtl import AnswerStatus
from tqdm import tqdm
EVALUATE_TIMES = 3
tag = '$tag'
evaluators = [load_registered_automatic_evaluator(evaluator_name='tooleval_local_judge', evaluators_cfg_path='toolbench/tooleval/evaluators')]
for group in ['G1_instruction', 'G2_instruction', 'G3_instruction']:
    cf = f'eval_converted_local/{tag}/{group}.json'
    if not os.path.exists(cf): continue
    examples = json.load(open(cf))
    sd = f'eval_results_local/{tag}'; os.makedirs(sd, exist_ok=True)
    sf = f'{sd}/{group}_results.json'
    results = json.load(open(sf)) if os.path.exists(sf) else {}
    for qid, ex in tqdm(examples.items(), desc=f'{tag} {group}'):
        if qid in results and len(results[qid].get('is_solved',{})) >= EVALUATE_TIMES: continue
        if qid not in results: results[qid] = {'query': ex['query'], 'is_solved': {}}
        for r in range(EVALUATE_TIMES):
            if str(r) in results[qid].get('is_solved',{}): continue
            fa = str(ex['answer'].get('final_answer',''))
            if 'Finish' not in fa and 'give_answer' not in fa:
                results[qid]['is_solved'][str(r)] = str(AnswerStatus.Unsolved); continue
            try:
                status, _ = random.choice(evaluators).check_is_solved({'query': ex['query'], 'available_tools': ex['available_tools']}, ex['answer'])
                results[qid]['is_solved'][str(r)] = str(status)
            except: results[qid]['is_solved'][str(r)] = str(AnswerStatus.Unsure)
        json.dump(results, open(sf, 'w'), ensure_ascii=False, indent=2)
    passed = sum(1 for r in results.values() if sum(1 for v in r.get('is_solved',{}).values() if 'Solved' in v and 'Unsolved' not in v) > 1.5)
    print(f'{tag} {group}: {passed}/{len(results)} = {passed/len(results)*100:.1f}%')
"
    echo "[$(date '+%H:%M:%S')] Done $tag"
done

# ---- Cleanup ----
kill $VLLM_PID 2>/dev/null
echo "================================================"
echo "ALL EVALUATIONS DONE! $(date)"
echo "================================================"