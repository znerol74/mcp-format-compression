# MCP Format Compression Experiment Pipeline

Research project comparing JSON vs TOON vs TRON serialization formats for MCP (Model Context Protocol) tool schemas, tool call outputs, and tool results. Measures the effect of token compression on LLM tool-calling accuracy across four benchmarks: two MCP-based (MCPToolBenchPP, MCP-Universe) and two function-calling (BFCL, StableToolBench).

## Project Structure

```
phd/
├── run_experiments.py       # Unified experiment runner (MCPToolBenchPP + MCP-Universe)
├── token_analysis.py        # Exp0: Offline token counting (no API calls)
├── analyze_results.py       # Post-hoc result analysis + LaTeX table generation
├── CLAUDE.md                # This file
│
├── shared_format/           # Shared serialization library (used by all benchmarks)
│   ├── src/shared_format/
│   │   ├── __init__.py      # Re-exports: serialize, deserialize, deserialize_lenient, ToolFormat, count_tokens
│   │   ├── converter.py     # Core: ToolFormat enum, serialize(), deserialize()
│   │   ├── prompt_snippets.py  # serialize_tools(), format explanations/intros/reminders
│   │   └── token_counter.py    # count_tokens() using tiktoken cl100k_base
│   └── pyproject.toml
│
├── MCPToolBenchPP/          # Benchmark 1: Single-turn MCP tool calling (submodule)
│   ├── run.py               # CLI entry: --tool_format, --tool_call_format, --model, --category
│   ├── src/mcp_tool_bench/
│   │   ├── format_converter.py  # ToolFormatConverter with decoupled input/output formats
│   │   └── agents/base_tool_call_agent/run_tool_call.py  # Main benchmark loop
│   └── data/                # Task files per category (finance, pay, search, file_system, map, browser)
│
├── MCP-Universe/            # Benchmark 2: Multi-turn MCP ReAct agent (submodule)
│   ├── run_full_test.py     # CLI entry: takes YAML config path
│   ├── mcpuniverse/
│   │   ├── agent/
│   │   │   ├── base.py          # BaseAgentConfig with tool_format + tool_call_format
│   │   │   ├── format_converter.py  # ToolFormatConverter with decoupled formats
│   │   │   └── react.py        # ReAct agent: builds prompts, parses responses, calls tools
│   │   ├── benchmark/configs/mcpuniverse/test_full/  # YAML configs per category x format
│   │   └── mcp/
│   │       ├── configs/server_list.json  # MCP server definitions
│   │       └── client.py       # Server subprocess launcher
│   └── .env                 # API keys (OPENROUTER_API_KEY, SERPAPI_KEY, etc.)
│
├── gorilla/                 # Benchmark 3: BFCL v4 function calling (submodule)
│   └── berkeley-function-call-leaderboard/
│       ├── bfcl_eval/
│       │   ├── constants/
│       │   │   ├── model_config.py      # Model registry (our models: qwen3-30b-local, -toon, -tron, -full variants)
│       │   │   ├── default_prompts.py   # OUTPUT_FORMAT_MAPPING (added toon/tron entries)
│       │   │   └── enums.py             # ReturnFormat enum (added TOON, TRON)
│       │   ├── model_handler/
│       │   │   ├── utils.py             # format_function_doc() + ast_parse() (added toon/tron branches)
│       │   │   └── api_inference/
│       │   │       └── openai_completion.py  # OpenWebUICompletionsHandler for local model
│       │   ├── eval_checker/eval_runner.py   # AST eval (deterministic, no LLM judge)
│       │   └── data/                    # BFCL_v4_*.json test files
│       ├── result/                      # Generated results per model
│       ├── score/                       # Evaluation scores per model
│       └── .venv/                       # Separate venv
│
├── StableToolBench/         # Benchmark 4: Multi-turn function calling (replaces ToolBench)
│   ├── toolbench/
│   │   ├── inference/
│   │   │   ├── qa_pipeline.py           # Main entry: --tool_format, --tool_call_format, --openwebui_model
│   │   │   ├── qa_pipeline_multithread.py # Multi-threaded variant
│   │   │   ├── LLM/
│   │   │   │   ├── openwebui_model.py   # OpenWebUIReACT: format-aware ReACT model (our addition)
│   │   │   │   └── chatgpt_function_model.py  # OpenAI function calling model
│   │   │   ├── Downstream_tasks/
│   │   │   │   └── rapidapi.py          # rapidapi_wrapper with _deserialize/_serialize (our addition)
│   │   │   └── Algorithms/
│   │   │       └── single_chain.py      # CoT ReACT loop
│   │   ├── tooleval/
│   │   │   ├── eval_pass_rate.py        # Solvable Pass Rate (SoPR) via GPT-4 judge
│   │   │   └── eval_preference.py       # Solvable Win Rate (SoWR) pairwise comparison
│   │   └── utils.py                     # process_system_message() with format support
│   ├── server/
│   │   ├── main.py                      # Virtual API server (GPT-based cache)
│   │   ├── config.yml                   # Server config (OpenAI key, cache path)
│   │   ├── tools/                       # 7,000+ API tool definitions
│   │   └── tool_response_cache/         # Cached API responses (no RapidAPI key needed)
│   ├── solvable_queries/test_instruction/  # Pre-filtered solvable queries
│   │   ├── G1_instruction.json          # 163 queries, single-tool
│   │   ├── G2_instruction.json          # 106 queries, intra-category multi-tool
│   │   └── G3_instruction.json          # 61 queries, cross-category multi-tool
│   └── .venv/                           # Separate venv (CPU-only torch)
│
├── toon-python/             # TOON format library (submodule, pip install -e)
└── tron-python/             # TRON format library (submodule, pip install -e)
```

## Benchmarks Overview

| | MCPToolBenchPP | MCP-Universe | BFCL v4 | StableToolBench |
|---|---|---|---|---|
| **Type** | MCP single-turn | MCP multi-turn ReAct | Function calling | Multi-turn ReAct |
| **Tasks** | ~1,509 | ~231 | ~4,400 | 330 (solvable) |
| **Tools** | MCP servers | MCP servers | Synthetic + live APIs | 7,000+ real APIs (cached) |
| **LLM output** | JSON/TOON/TRON tool calls | JSON/TOON/TRON tool calls | Python/JSON/XML/TOON/TRON | ReACT (Thought/Action/Action Input) |
| **Eval method** | LLM-as-judge | Task completion | AST matching (deterministic) | GPT-4 judge (SoPR + SoWR) |
| **Token tracking** | Via API response | Via API response | input_token + output_token per task | total_tokens per query |
| **Format integration** | shared_format (ToolFormatConverter) | shared_format (ToolFormatConverter) | format_function_doc() + ast_parse() | OpenWebUIReACT + rapidapi_wrapper |
| **exp1 support** | Yes | Yes | Yes (func_doc_fmt) | Yes (schema serialization) |
| **exp2 support** | Yes | Yes | Yes, non-live + irrelevance only | Yes (schema + Action Input + observation) |

## Formats

| Format | Description | Library |
|--------|-------------|---------|
| **JSON** | Standard `json.dumps(indent=2)` | stdlib |
| **TOON** | Token-Oriented Object Notation -- indentation-based, no quotes/braces | `toon-python` (v0.9.0b1) |
| **TRON** | Compact JSON with class definitions for repeated structures | `tron-python` (v0.1.0) |

All serialization goes through `shared_format.serialize()` / `shared_format.deserialize()`.

## Experiments

| Experiment | Description | Cost |
|------------|-------------|------|
| **exp0** | Offline token counting across formats (no API calls) | Free |
| **exp1** | Input compression: schemas + results in TOON/TRON, tool calls in JSON/Python | ~$50 |
| **exp2** | Full compression: everything (schemas, results, AND tool calls) compressed | ~$50 |
| **exp3** | Model comparison: exp2 with multiple models | ~$50/model |

### Format Scoping

| Scope | Schema format | Result format | Tool call format |
|-------|--------------|---------------|------------------|
| `input` (exp1) | compressed | compressed | **JSON/Python** |
| `full` (exp2) | compressed | compressed | **compressed** |

When format=json, both scopes are identical (baseline).

## How to Run

### Prerequisites

Each benchmark has its own venv. Install dependencies:

```bash
# shared_format (installed as editable in all venvs)
cd shared_format && pip install -e .

# MCPToolBenchPP
cd MCPToolBenchPP && pip install -r requirements.txt
pip install -e ../shared_format -e ../toon-python -e ../tron-python

# MCP-Universe
cd MCP-Universe && pip install -r requirements.txt
pip install -e ../shared_format -e ../toon-python -e ../tron-python

# BFCL
cd gorilla/berkeley-function-call-leaderboard
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install -e ../../shared_format -e ../../toon-python -e ../../tron-python

# StableToolBench (already set up with CPU-only torch)
cd StableToolBench
# venv already created with: openai, requests, torch (CPU), transformers, shared_format, toon, tron
# Cache and tools already downloaded from HuggingFace
```

### API Keys

Set in environment or `.env` files:
- `OPENROUTER_API_KEY` -- for LLM calls via OpenRouter (qwen3-32b, gpt-4o-mini, etc.)
- `OPENWEBUI_API_KEY` -- for local model via OpenWebUI (qwen3:30b-instruct)
- `OPENWEBUI_BASE_URL` -- `https://openwebui.know.know-center.at/api`
- `SERPAPI_KEY` -- for MCP-Universe web_search tasks
- `GOOGLE_MAPS_API_KEY` -- for MCP-Universe location_navigation tasks
- `OPENAI_API_KEY` -- for StableToolBench virtual API server (cache miss fallback only)

### Running Experiments

```bash
# Offline token analysis (free)
python run_experiments.py --exp exp0

# Full compression, MCP benchmarks, qwen3-32b
python run_experiments.py --exp exp2 --model qwen3-32b --benchmark benchpp universe

# Input-only compression, single benchmark
python run_experiments.py --exp exp1 --model qwen3-32b --benchmark benchpp

# Specific formats only
python run_experiments.py --exp exp2 --formats toon tron --benchmark universe

# Model comparison
python run_experiments.py --exp exp3 --benchmark benchpp universe
```

### Running Benchmarks Individually

**MCPToolBenchPP:**
```bash
cd MCPToolBenchPP
.venv/bin/python3 run.py \
  --stage tool_call \
  --input_file ./data/finance/finance_0724_single_v3.json \
  --category finance \
  --model qwen3-32b \
  --tool_mode prompt \
  --tool_format toon \
  --tool_call_format json \
  --llm_as_judge_model deepseek-v3.2 \
  --pass_k 1 \
  --evaluation_trial_per_task 5
```

**MCP-Universe:**
```bash
cd MCP-Universe
# Must prepend venv to PATH so MCP servers find their dependencies
PATH=".venv/bin:$PATH" .venv/bin/python3 run_full_test.py \
  mcpuniverse/benchmark/configs/mcpuniverse/test_full/financial_analysis_toon.yaml
```

**BFCL (Gorilla):**
```bash
cd gorilla/berkeley-function-call-leaderboard

# Baseline (JSON input, Python output)
OPENWEBUI_API_KEY=... OPENWEBUI_BASE_URL=... \
.venv/bin/bfcl generate --model qwen3-30b-local --test-category simple_python --num-threads 1

# exp1: TOON input, Python output
BFCL_PROMPT_FORMAT_OVERRIDE="ret_fmt=python&tool_call_tag=False&func_doc_fmt=toon&prompt_fmt=plaintext&style=classic" \
.venv/bin/bfcl generate --model qwen3-30b-local-toon --test-category simple_python --num-threads 1

# exp2: TOON input + TOON output
BFCL_PROMPT_FORMAT_OVERRIDE="ret_fmt=toon&tool_call_tag=False&func_doc_fmt=toon&prompt_fmt=plaintext&style=classic" \
.venv/bin/bfcl generate --model qwen3-30b-local-toon-full --test-category simple_python --num-threads 1

# Evaluate
.venv/bin/bfcl evaluate --model qwen3-30b-local --test-category simple_python
```

**StableToolBench:**

IMPORTANT: The virtual API server must be running before any StableToolBench experiment.

```bash
# Terminal 1: Start the virtual API server (keep running)
cd StableToolBench/server
../.venv/bin/python3 main.py
# Runs on http://localhost:8080/virtual
# Uses cached responses; falls back to GPT-4 Turbo for cache misses
# OpenAI key configured in server/config.yml

# Terminal 2: Run inference
cd StableToolBench
export PYTHONPATH=./:./toolbench/inference
export SERVICE_URL="http://localhost:8080/virtual"
export OPENWEBUI_API_KEY=...
export OPENWEBUI_BASE_URL=https://openwebui.know.know-center.at/api

# Baseline (JSON)
.venv/bin/python3 toolbench/inference/qa_pipeline.py \
    --tool_root_dir server/tools \
    --backbone_model openwebui \
    --openwebui_model qwen3:30b-instruct \
    --tool_format json --tool_call_format json \
    --method CoT@1 \
    --input_query_file solvable_queries/test_instruction/G1_instruction.json \
    --output_answer_file results/json/G1_instruction

# exp1: TOON input, JSON tool calls
.venv/bin/python3 toolbench/inference/qa_pipeline.py \
    --tool_root_dir server/tools \
    --backbone_model openwebui \
    --openwebui_model qwen3:30b-instruct \
    --tool_format toon --tool_call_format json \
    --method CoT@1 \
    --input_query_file solvable_queries/test_instruction/G1_instruction.json \
    --output_answer_file results/toon_exp1/G1_instruction

# exp2: TOON full (schemas + tool calls + observations all TOON)
.venv/bin/python3 toolbench/inference/qa_pipeline.py \
    --tool_root_dir server/tools \
    --backbone_model openwebui \
    --openwebui_model qwen3:30b-instruct \
    --tool_format toon --tool_call_format toon \
    --method CoT@1 \
    --input_query_file solvable_queries/test_instruction/G1_instruction.json \
    --output_answer_file results/toon_exp2/G1_instruction
```

Via `run_experiments.py` (handles all groups and formats automatically):
```bash
# IMPORTANT: Start the virtual API server first!
cd StableToolBench/server && ../.venv/bin/python3 main.py &

# Then run experiments
python run_experiments.py --exp exp2 --benchmark toolbench
python run_experiments.py --exp exp1 --benchmark toolbench
```

### Analyzing Results

```bash
# Collect results from all benchmarks, generate tables
python analyze_results.py

# Output: experiment_logs/analysis/paper_tables.tex
#         experiment_logs/analysis/combined_results.json
```

## Key Design Decisions

1. **Decoupled input/output formats**: `tool_format` controls what the LLM reads (schemas, results), `tool_call_format` controls what the LLM writes (tool calls). This enables exp1 vs exp2 comparison.

2. **TRON serialize_tools() fix**: TRON class definitions only activate when repeated structures appear in a single `TRON.stringify()` call. `serialize_tools()` passes the full tool list as one call for TRON (enabling class definitions), but serializes individually for JSON/TOON (which don't benefit from batching).

3. **MCP-Universe PATH requirement**: MCP servers are launched as subprocesses using `shutil.which("python3")`. The venv `bin/` must be on PATH so servers find their dependencies (httpx, etc.).

4. **shared_format is the single source of truth**: All benchmarks delegate serialization to `shared_format`. Changes to format handling only need to happen in one place.

5. **BFCL format override mechanism**: `BFCL_PROMPT_FORMAT_OVERRIDE` env var overrides the per-entry format parameters (ret_fmt, func_doc_fmt, etc.) so we can run all tasks with a specific format without modifying test data files. Separate model registry entries (e.g. qwen3-30b-local-toon) keep results isolated.

6. **BFCL OpenWebUI handler**: `OpenWebUICompletionsHandler` subclass reads `OPENWEBUI_API_KEY` and `OPENWEBUI_BASE_URL` instead of standard OpenAI env vars, allowing both OpenRouter and local model to coexist.

7. **BFCL exp2 compatibility**: Multi-turn categories hard-code `ReturnFormat.PYTHON` in eval, so exp2 (compressed output) only works with non-live single-turn + irrelevance categories. Multi-turn runs exp1 only.

## Benchmark Categories

### MCP Benchmarks

| Domain | MCPToolBenchPP | MCP-Universe |
|--------|---------------|--------------|
| Web/Search | search (181 tasks) | web_search (55 tasks) |
| Browser | browser (187 tasks) | browser_automation (39 tasks) |
| Finance | finance (90 tasks) | financial_analysis (40 tasks) |
| Maps/Location | map (500 tasks) | location_navigation (45 tasks) |
| Files/Code | file_system (241 tasks) | repository_management (33 tasks) |
| Payments | pay (310 tasks) | -- |
| 3D Design | -- | 3d_design (19 tasks) |

### BFCL Categories

| Category | Tasks | Type | exp2 compatible |
|----------|-------|------|-----------------|
| simple_python | 400 | Single function call | Yes |
| simple_java | 100 | Single call (Java) | No |
| simple_javascript | 50 | Single call (JS) | No |
| multiple | 200 | Choose correct function | Yes |
| parallel | 200 | Multiple simultaneous calls | Yes |
| parallel_multiple | 200 | Parallel + selection | Yes |
| irrelevance | 240 | Refuse (no relevant function) | Yes |
| live_simple | 258 | Real API, single call | Yes |
| live_multiple | 1,053 | Real API, selection | Yes |
| live_parallel | 16 | Real API, parallel | Yes |
| live_parallel_multiple | 24 | Real API, parallel + selection | Yes |
| live_irrelevance | 884 | Real API, refuse | Yes |
| live_relevance | 16 | Confirm relevance | Yes |
| multi_turn_base | 200 | Multi-step with results | No (exp1 only) |
| multi_turn_miss_func | 200 | No functions, should refuse | No (exp1 only) |
| multi_turn_miss_param | 200 | Missing param, should ask | No (exp1 only) |
| multi_turn_long_context | 200 | Large context multi-turn | No (exp1 only) |

### StableToolBench Groups

| Group | Description | Solvable queries | Avg tools/query |
|-------|-------------|-----------------|-----------------|
| G1_instruction | Single-tool, unseen instructions | 163 | 5.3 |
| G2_instruction | Intra-category multi-tool | 106 | 6.5 |
| G3_instruction | Cross-category multi-tool (hardest) | 61 | 5.8 |

Total: 330 solvable queries across 7,000+ cached real APIs from 49 categories.
Virtual API server uses cached responses (no RapidAPI key needed). OpenAI key only for rare cache misses.

## BFCL Model Registry

| Registry key | Model | Handler | Format scope |
|---|---|---|---|
| qwen3-30b-local | qwen3:30b-instruct | OpenWebUICompletionsHandler | Baseline (JSON/Python) |
| qwen3-30b-local-toon | qwen3:30b-instruct | OpenWebUICompletionsHandler | exp1: TOON input, Python output |
| qwen3-30b-local-tron | qwen3:30b-instruct | OpenWebUICompletionsHandler | exp1: TRON input, Python output |
| qwen3-30b-local-json-full | qwen3:30b-instruct | OpenWebUICompletionsHandler | exp2 baseline: JSON input + JSON output |
| qwen3-30b-local-toon-full | qwen3:30b-instruct | OpenWebUICompletionsHandler | exp2: TOON input + TOON output |
| qwen3-30b-local-tron-full | qwen3:30b-instruct | OpenWebUICompletionsHandler | exp2: TRON input + TRON output |
| qwen3-32b-openrouter | qwen/qwen3-32b | OpenAICompletionsHandler | Any (via OpenRouter) |

## Models

| Model | Endpoint | Name | Cost (input/output per M tokens) |
|-------|----------|------|----------------------------------|
| Qwen3-32b | OpenRouter | qwen3-32b | $0.20 / $0.60 |
| Qwen3-30b | OpenWebUI (local) | qwen3-30b-local | Free |
| GPT-4o-mini | OpenRouter | gpt-4o-mini | $0.15 / $0.60 |
| DeepSeek V3 | OpenRouter | deepseek-v3 | $0.30 / $0.88 |

## Monitoring Running Experiments

```bash
# MCP benchmarks
tail -20 experiment_logs/exp2_benchpp_run.log
tail -20 experiment_logs/exp2_universe_run.log
grep -c "OK\|FAIL" experiment_logs/exp2_benchpp_run.log

# BFCL pipeline
tail -f experiment_logs/bfcl_pipeline.log
find gorilla/berkeley-function-call-leaderboard/result -name "*result.json" \
  -not -path "*/memory*" -type f \
  -exec sh -c 'echo "$(wc -l < "$1") $1"' _ {} \; | sort

# Check processes
ps aux | grep -E "run_experiments|bfcl" | grep -v grep
```

## Result Locations

- **MCPToolBenchPP logs**: `MCPToolBenchPP/logs/{category}/{filename}.json`
- **MCP-Universe logs**: `MCP-Universe/log/test_full/{name}.log`
- **MCP-Universe reports**: `MCP-Universe/test_full_report/{name}`
- **BFCL results**: `gorilla/berkeley-function-call-leaderboard/result/{model}/{category}/`
- **BFCL scores**: `gorilla/berkeley-function-call-leaderboard/score/{model}/`
- **StableToolBench results**: `StableToolBench/results/{tag}_{fmt}_{group}_{timestamp}/`
- **Experiment runner logs**: `experiment_logs/benchpp_exp2_*.log`, `experiment_logs/universe_exp2_*.log`
- **Token analysis CSVs**: `experiment_logs/token_analysis/*.csv`
- **Analysis output**: `experiment_logs/analysis/paper_tables.tex`
