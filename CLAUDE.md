# MCP Format Compression Experiment Pipeline

Research project comparing JSON vs TOON vs TRON serialization formats for MCP (Model Context Protocol) tool schemas, tool call outputs, and tool results. Measures the effect of token compression on LLM tool-calling accuracy across two benchmarks.

## Project Structure

```
phd/
├── run_experiments.py       # Unified experiment runner (main entry point)
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
├── MCPToolBenchPP/          # Benchmark 1: Single-turn tool calling (submodule)
│   ├── run.py               # CLI entry: --tool_format, --tool_call_format, --model, --category
│   ├── src/mcp_tool_bench/
│   │   ├── format_converter.py  # ToolFormatConverter with decoupled input/output formats
│   │   └── agents/base_tool_call_agent/run_tool_call.py  # Main benchmark loop
│   └── data/                # Task files per category (finance, pay, search, file_system, map, browser)
│
├── MCP-Universe/            # Benchmark 2: Multi-turn ReAct agent (submodule)
│   ├── run_full_test.py     # CLI entry: takes YAML config path
│   ├── mcpuniverse/
│   │   ├── agent/
│   │   │   ├── base.py          # BaseAgentConfig with tool_format + tool_call_format
│   │   │   ├── format_converter.py  # ToolFormatConverter with decoupled formats
│   │   │   └── react.py        # ReAct agent: builds prompts, parses responses, calls tools
│   │   ├── benchmark/configs/mcpuniverse/test_full/  # YAML configs per category × format
│   │   └── mcp/
│   │       ├── configs/server_list.json  # MCP server definitions
│   │       └── client.py       # Server subprocess launcher
│   └── .env                 # API keys (OPENROUTER_API_KEY, SERPAPI_KEY, etc.)
│
├── toon-python/             # TOON format library (submodule, pip install -e)
└── tron-python/             # TRON format library (submodule, pip install -e)
```

## Formats

| Format | Description | Library |
|--------|-------------|---------|
| **JSON** | Standard `json.dumps(indent=2)` | stdlib |
| **TOON** | Token-Oriented Object Notation — indentation-based, no quotes/braces | `toon-python` (v0.9.0b1) |
| **TRON** | Compact JSON with class definitions for repeated structures | `tron-python` (v0.1.0) |

All serialization goes through `shared_format.serialize()` / `shared_format.deserialize()`.

## Experiments

| Experiment | Description | Cost |
|------------|-------------|------|
| **exp0** | Offline token counting across formats (no API calls) | Free |
| **exp1** | Input compression: schemas + results in TOON/TRON, tool calls in JSON | ~$50 |
| **exp2** | Full compression: everything (schemas, results, AND tool calls) compressed | ~$50 |
| **exp3** | Model comparison: exp2 with multiple models | ~$50/model |

### Format Scoping

| Scope | Schema format | Result format | Tool call format |
|-------|--------------|---------------|------------------|
| `input` (exp1) | compressed | compressed | **JSON** |
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
```

### API Keys

Set in environment or `.env` files:
- `OPENROUTER_API_KEY` — for LLM calls (qwen3-32b, gpt-4o-mini, etc.)
- `SERPAPI_KEY` — for MCP-Universe web_search tasks
- `GOOGLE_MAPS_API_KEY` — for MCP-Universe location_navigation tasks

### Running Experiments

```bash
# Offline token analysis (free)
python run_experiments.py --exp exp0

# Full compression, both benchmarks, qwen3-32b
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

## Benchmark Categories

| Domain | MCPToolBenchPP | MCP-Universe |
|--------|---------------|--------------|
| Web/Search | search (181 tasks) | web_search (55 tasks) |
| Browser | browser (187 tasks) | browser_automation (39 tasks) |
| Finance | finance (90 tasks) | financial_analysis (40 tasks) |
| Maps/Location | map (500 tasks) | location_navigation (45 tasks) |
| Files/Code | file_system (241 tasks) | repository_management (33 tasks) |
| Payments | pay (310 tasks) | — |
| 3D Design | — | 3d_design (19 tasks) |

## Models

| Model | OpenRouter ID | MCPToolBenchPP name | Cost (input/output per M tokens) |
|-------|---------------|---------------------|----------------------------------|
| Qwen3-32b | qwen/qwen3-32b | qwen3-32b | $0.20 / $0.60 |
| GPT-4o-mini | openai/gpt-4o-mini | gpt-4o-mini | $0.15 / $0.60 |
| DeepSeek V3 | deepseek/deepseek-chat-v3-0324 | deepseek-v3 | $0.30 / $0.88 |

## Monitoring Running Experiments

```bash
# Check progress (completed categories)
tail -20 experiment_logs/exp2_benchpp_run.log
tail -20 experiment_logs/exp2_universe_run.log

# Count completed runs
grep -c "OK\|FAIL" experiment_logs/exp2_benchpp_run.log

# Check processes are alive
ps aux | grep run_experiments | grep -v grep

# Check latest log sizes
ls -lt experiment_logs/benchpp_exp2_* | head -5
ls -lt experiment_logs/universe_exp2_* | head -5
```

## Result Locations

- **MCPToolBenchPP logs**: `MCPToolBenchPP/logs/{category}/{filename}.json`
- **MCP-Universe logs**: `MCP-Universe/log/test_full/{name}.log`
- **MCP-Universe reports**: `MCP-Universe/test_full_report/{name}`
- **Experiment runner logs**: `experiment_logs/benchpp_exp2_*.log`, `experiment_logs/universe_exp2_*.log`
- **Token analysis CSVs**: `experiment_logs/token_analysis/*.csv`
- **Analysis output**: `experiment_logs/analysis/paper_tables.tex`
