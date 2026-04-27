# Experiment Pipeline Status

*Last updated: 2026-03-26*

## Complete Results — exp2 (full compression)

### StableToolBench (multi-turn, 330 tasks, 7000+ APIs)

SoPR = Solvable Pass Rate, confirmed by GPT-4o-mini judge.

| Model | JSON | TOON | TRON |
|---|---|---|---|
| qwen3-30b (local) | 34.2% | 53.9% | **59.7%** |
| qwen3.5-35b (local) | 31.8% | 31.8% | 24.2% |
| qwen3-32b-FP16 (cluster) | running | - | - |
| qwen3-32b-AWQ (cluster) | queued | - | - |
| qwen3.5-122B-GPTQ (cluster) | queued | - | - |

Token savings (qwen3-30b):

| Component | JSON | TOON | TRON |
|---|---|---|---|
| Prompt | 4,773,429 | 3,154,615 (-33.9%) | 3,142,789 (-34.2%) |
| Completion | 311,278 | 107,182 (-65.6%) | 123,679 (-60.3%) |
| Total | 5,084,707 | 3,261,797 (-35.9%) | 3,266,468 (-35.8%) |
| Avg per task | 15,408 | 9,884 | 9,898 |
| Avg steps | 10.4 | 9.5 | 9.4 |

### MCPToolBenchPP (single-turn MCP, 5 categories)

| Model | Metric | JSON | TOON | TRON |
|---|---|---|---|---|
| qwen3-30b | pass@1 | 0.548 | 0.360 | 0.499 |
| qwen3-30b | tool_pass | 0.987 | 0.985 | 0.984 |
| qwen3-30b | param_pass | 0.872 | 0.632 | 0.836 |
| qwen3-32b | pass@1 | 0.594 | 0.333 | 0.471 |
| qwen3-32b | tool_pass | 0.940 | 0.964 | 0.963 |
| qwen3-32b | param_pass | 0.848 | 0.676 | 0.870 |

Per-category (qwen3-30b):

| Category | JSON | TOON | TRON |
|---|---|---|---|
| finance | 0.236 | 0.000 | 0.062 |
| pay | 0.511 | 0.218 | 0.472 |
| search | 0.849 | 0.665 | 0.803 |
| file_system | 0.882 | 0.700 | 0.825 |
| browser | 0.036 | 0.035 | 0.037 |

### MCP-Universe (multi-turn MCP, 176 tasks)

| Model | JSON | TOON | TRON |
|---|---|---|---|
| qwen3-32b | 13.1% | 4.0% | 12.5% |
| qwen3-30b | 3.4% | 2.8% | 0.0% |

Token usage (qwen3-32b):

| Metric | JSON | TOON | TRON |
|---|---|---|---|
| Total tokens | 36,829,866 | 39,479,610 (+7.2%) | 25,068,509 (-31.9%) |
| Avg iterations | 10.4 | 14.2 (+35.9%) | 8.4 (-19.5%) |
| Cost USD | $8.11 | $8.00 (-1.3%) | $5.55 (-31.5%) |

### BFCL (single-turn function calling, 1240 tasks)

| Model | JSON | TOON | TRON |
|---|---|---|---|
| qwen3-30b (exp2) | 95.0% | 55.0% | 94.4% |
| qwen3-30b (exp1) | 91.7% | 84.4% | 81.0% |

### Token Savings Summary

| Benchmark | TOON vs JSON | TRON vs JSON | Source |
|---|---|---|---|
| StableToolBench total | -35.9% | -35.8% | qwen3-30b |
| StableToolBench prompt | -33.9% | -34.2% | qwen3-30b |
| StableToolBench completion | -65.6% | -60.3% | qwen3-30b |
| MCP-Universe total | +7.2% (cascade!) | -31.9% | qwen3-32b |
| MCP-Universe iterations | +35.9% | -19.5% | qwen3-32b |
| MCPToolBenchPP schema | -43.1% | -11.1% | qwen3-32b |
| BFCL total | +3.2% | -0.4% | qwen3-30b |

## Key Findings

1. **TRON is best format overall**: preserves accuracy, saves tokens across all benchmarks
2. **TOON is unreliable**: helps on StableToolBench but fails on BFCL (parallel=0%) and finance tasks
3. **Compression benefit is model-dependent**: qwen3-30b gains +25.5% SoPR with TRON, qwen3.5-35b loses -7.6%
4. **Compression benefit scales with context pressure**: BFCL (tiny schemas) no benefit, StableToolBench (huge API docs) massive benefit
5. **Multi-turn cascading**: TRON reduces iterations (10.4→8.4), TOON increases them (10.4→14.2)
6. **Tool selection is format-agnostic**: ~98% across all formats on MCPToolBenchPP
7. **Parameter generation is the bottleneck**: TOON drops param_pass by 25%, TRON only 4%
8. **Completion tokens drop 60-66%** with compressed formats on StableToolBench

## Running Pipelines

### Cluster (VSC5)

| Job | Model | Partition | Status |
|---|---|---|---|
| 7537563 | Qwen3-32B-AWQ | 2x A40 | queued |
| 7538087 | Qwen3-32B-FP16 | 2x A100 | running (G1/json ~task 150/163, thinking enabled) |
| 7538301 | Qwen3.5-122B-GPTQ | 2x A100 | queued |

### Local

MCPToolBenchPP 30b v3 rerun: DONE (all 15 runs complete)
StableToolBench eval: DONE for qwen3-30b and qwen3.5

## Results Archive

All results in `/home/lkutschka/phd/results_archive/` organized by benchmark/model/experiment.
Cluster results on `/gpfs/data/fs73109/lkutschka/phd_results/` and `/gpfs/data/fs73109/lkutschka/phd_logs/`.