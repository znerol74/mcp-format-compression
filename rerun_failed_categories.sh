#!/bin/bash
# Re-run browser (all 3 formats) and search (toon + tron) on OpenRouter qwen3-32b
# Browser fix: DISPLAY=:99 set in mcp_config.json
# Search fix: new Tavily API key set in mcp_config.json

set -e
cd /home/lkutschka/phd/MCPToolBenchPP
PYTHON=".venv/bin/python3"
MODEL="qwen3-32b"
JUDGE="deepseek-v3.2"

run_one() {
    local cat=$1 fmt=$2 input=$3
    echo "$(date '+%H:%M:%S') Starting $cat $fmt..."
    $PYTHON run.py \
        --stage tool_call \
        --input_file "$input" \
        --category "$cat" \
        --model "$MODEL" \
        --tool_mode prompt \
        --tool_format "$fmt" \
        --tool_call_format "$fmt" \
        --llm_as_judge_model "$JUDGE" \
        --pass_k 1 \
        --evaluation_trial_per_task 5
    echo "$(date '+%H:%M:%S') Done $cat $fmt (exit=$?)"
}

# Browser: all 3 formats
run_one browser json ./data/browser/browser_0724_single_v3.json
run_one browser toon ./data/browser/browser_0724_single_v3.json
run_one browser tron ./data/browser/browser_0724_single_v3.json

# Search: all 3 formats (JSON baseline also had 432 quota errors)
run_one search json ./data/search/search_0725_single_v2.json
run_one search toon ./data/search/search_0725_single_v2.json
run_one search tron ./data/search/search_0725_single_v2.json

echo "All re-runs complete!"
