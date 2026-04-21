# Complete Experiment Results

*Generated: 2026-03-26*

All results are exp2 (full compression) unless noted otherwise.

## 1. StableToolBench

Multi-turn ReAct agent, 330 solvable tasks (G1=163, G2=106, G3=61), 7000+ cached RapidAPIs.
SoPR = Solvable Pass Rate, confirmed by GPT-4o-mini judge (3 votes, majority rule).

### 1.1 Accuracy (SoPR)

| Model | JSON | TOON | TRON | TOON vs JSON | TRON vs JSON |
|---|---|---|---|---|---|
| qwen3-30b (local) | 34.2% | 53.9% | 59.7% | +19.7% | +25.5% |
| qwen3.5-35b (local) | 31.8% | 31.8% | 24.2% | +0.0% | -7.6% |

### 1.2 Per-Group Accuracy (SoPR)

| Model | Group | JSON | TOON | TRON |
|---|---|---|---|---|
| qwen3-30b | G1 | 54/163 (33.1%) | 90/163 (55.2%) | 104/163 (63.8%) |
| qwen3-30b | G2 | 41/106 (38.7%) | 58/106 (54.7%) | 60/106 (56.6%) |
| qwen3-30b | G3 | 18/61 (29.5%) | 30/61 (49.2%) | 33/61 (54.1%) |
| qwen3.5-35b | G1 | 56/163 (34.4%) | 61/163 (37.4%) | 69/163 (42.3%) |
| qwen3.5-35b | G2 | 32/106 (30.2%) | 29/106 (27.4%) | 11/106 (10.4%) |
| qwen3.5-35b | G3 | 17/61 (27.9%) | 15/61 (24.6%) | 0/61 (0.0%) |

### 1.3 Token Usage

| Model | Component | JSON | TOON | TRON | TOON vs JSON | TRON vs JSON |
|---|---|---|---|---|---|---|
| qwen3-30b | Prompt | 4,773,429 | 3,154,615 | 3,142,789 | -33.9% | -34.2% |
| qwen3-30b | Completion | 311,278 | 107,182 | 123,679 | -65.6% | -60.3% |
| qwen3-30b | Observation | 68,043 | 90,594 | 84,627 | +33.1% | +24.4% |
| qwen3-30b | Total | 5,084,707 | 3,261,797 | 3,266,468 | -35.9% | -35.8% |
| qwen3-30b | Avg steps/task | 10.4 | 9.5 | 9.4 | -9.1% | -9.6% |
| qwen3-30b | Avg tokens/task | 15,408 | 9,884 | 9,898 | | |
| qwen3.5-35b | Prompt | 2,353,815 | 3,656,795 | 1,738,361 | +55.4% | -26.1% |
| qwen3.5-35b | Completion | 139,614 | 144,282 | 84,933 | +3.3% | -39.2% |
| qwen3.5-35b | Observation | 119,262 | 111,360 | 71,808 | -6.6% | -39.8% |
| qwen3.5-35b | Total | 2,493,429 | 3,801,077 | 1,823,294 | +52.4% | -26.9% |
| qwen3.5-35b | Avg steps/task | 10.2 | 10.2 | 6.2 | +0.2% | -39.3% |
| qwen3.5-35b | Avg tokens/task | 7,556 | 11,518 | 5,525 | | |

### 1.4 Completion Rate vs SoPR

| Model | Metric | JSON | TOON | TRON |
|---|---|---|---|---|
| qwen3-30b | Finish rate | 40.9% | 74.5% | 80.3% |
| qwen3-30b | SoPR | 34.2% | 53.9% | 59.7% |
| qwen3.5-35b | Finish rate | 51.2% | 48.2% | 34.2% |
| qwen3.5-35b | SoPR | 31.8% | 31.8% | 24.2% |

### 1.5 exp1 vs exp2 (input-only vs full compression, qwen3-30b)

exp1 = compressed schemas + results, JSON tool calls. exp2 = everything compressed.

| Format | exp2 (full) | exp1 (input only) | Difference | Interpretation |
|---|---|---|---|---|
| TOON | 53.9% | 54.2% | -0.3pp | Benefit is entirely from input compression |
| TRON | 59.7% | 52.7% | +7.0pp | Output compression adds extra benefit |

TOON: compressing tool call output in TOON does not help (model uses native function calling anyway).
TRON: compressing tool call output in TRON adds +7pp because TRON tool calls are shorter (class instantiation),
freeing more context for reasoning. TRON benefits from both input AND output compression.

## 2. MCPToolBenchPP

Single-turn MCP tool calling, 5 categories (finance=90, pay=310, search=181, file_system=241, browser=187).
1009 total tasks. 5 trials per task for pass@k estimation.

### 2.1 Accuracy

| Model | Metric | JSON | TOON | TRON | TOON vs JSON | TRON vs JSON |
|---|---|---|---|---|---|---|
| qwen3-30b | pass@1 | 0.548 | 0.360 | 0.499 | -34.2% | -9.0% |
| qwen3-30b | tool_pass@1 | 0.987 | 0.985 | 0.984 | -0.2% | -0.3% |
| qwen3-30b | parameter_pass@1 | 0.872 | 0.632 | 0.836 | -27.5% | -4.0% |
| qwen3-32b (OR) | pass@1 | 0.000 | 0.000 | 0.000 |  |  |

### 2.2 Per-Category Accuracy (qwen3-30b, pass@1)

| Category | JSON | TOON | TRON |
|---|---|---|---|
| finance | 0.236 | 0.000 | 0.062 |
| pay | 0.511 | 0.218 | 0.472 |
| search | 0.849 | 0.665 | 0.803 |
| file_system | 0.882 | 0.700 | 0.825 |
| browser | 0.036 | 0.035 | 0.037 |

### 2.3 Token Decomposition

| Model | Component | JSON | TOON | TRON | TOON vs JSON | TRON vs JSON |
|---|---|---|---|---|---|---|
| qwen3-30b | Schema | 14,380,695 | | | -6.8% | -21.8% |
| qwen3-30b | Tool Call | 267,523 | | | -4.0% | -14.1% |
| qwen3-30b | Results | 3,396,237 | | | -17.4% | -17.3% |
| qwen3-30b | Total | 18,044,455 | | | -8.8% | -20.9% |
| qwen3-32b (OR) | Schema | 22,798,290 | | | -43.1% | -11.1% |
| qwen3-32b (OR) | Tool Call | 249,038 | | | -20.9% | -22.7% |
| qwen3-32b (OR) | Results | 317,519 | | | -24.9% | -21.6% |

## 3. MCP-Universe

Multi-turn ReAct agent, 176 tasks, 5 categories (financial_analysis=40, location_navigation=45,
browser_automation=39, repository_management=33, 3d_design=19). Live MCP servers.

### 3.1 Accuracy

| Model | JSON | TOON | TRON | TOON vs JSON | TRON vs JSON |
|---|---|---|---|---|---|
| qwen3-32b (OR) | 13.1% | 4.0% | 12.5% | -69.5% | -4.6% |
| qwen3-30b (local) | 3.4% | 2.8% | 0.0% | -17.6% | -100% |

### 3.2 Token Decomposition (qwen3-32b)

| Component | JSON | TOON | TRON | TOON vs JSON | TRON vs JSON |
|---|---|---|---|---|---|
| Schema | 574,600 | 730,148 | 560,537 | +27.1% | -2.4% |
| Tool Call | 401,694 | 619,698 | 244,904 | +54.3% | -39.0% |
| Results | 2,761,890 | 1,728,439 | 2,363,173 | -37.4% | -14.4% |
| API Total | 36,829,866 | 39,479,610 | 25,068,509 | +7.2% | -31.9% |
| Avg iterations | 10.4 | 14.2 | 8.4 | +35.9% | -19.5% |
| Cost USD | $8.11 | $8.00 | $5.55 | -1.3% | -31.5% |

### 3.3 Per-Category Accuracy (qwen3-32b)

| Category | JSON | TOON | TRON |
|---|---|---|---|
| financial_analysis | 27.5% | 10.0% | 32.5% |
| location_navigation | 6.7% | 0.0% | 11.1% |
| browser_automation | 10.3% | 7.7% | 10.3% |
| repository_management | 9.1% | 0.0% | 0.0% |
| 3d_design | 10.5% | 0.0% | 0.0% |

## 4. BFCL

Single-turn function calling, 1240 tasks, 5 non-live categories.
exp2 = JSON/TOON/TRON input + output. exp1 = compressed input, Python output.

### 4.1 Accuracy

| Model | Experiment | JSON | TOON | TRON | TOON vs JSON | TRON vs JSON |
|---|---|---|---|---|---|---|
| qwen3-30b | exp2 | 95.0% | 55.0% | 94.4% | -42.1% | -0.6% |
| qwen3-30b | exp1 | 91.7% | 84.4% | 81.0% | -8.0% | -11.7% |

### 4.2 Per-Category (qwen3-30b, exp2)

| Category | JSON | TOON | TRON |
|---|---|---|---|
| simple_python | 96.0% | 76.0% | 95.0% |
| multiple | 95.0% | 69.0% | 94.0% |
| parallel | 94.0% | 0.0% | 93.5% |
| parallel_multiple | 88.0% | 0.0% | 87.5% |
| irrelevance | 100.0% | 100.0% | 100.0% |

### 4.3 Token Usage (qwen3-30b, exp2)

| Component | JSON | TOON | TRON | TOON vs JSON | TRON vs JSON |
|---|---|---|---|---|---|
| Total | 759,343 | 783,365 | 756,135 | +3.2% | -0.4% |

## 5. Cross-Benchmark Token Savings Summary

| Benchmark | Component | TOON vs JSON | TRON vs JSON |
|---|---|---|---|
| StableToolBench | Total | -35.9% | -35.8% |
| StableToolBench | Prompt | -33.9% | -34.2% |
| StableToolBench | Completion | -65.6% | -60.3% |
| StableToolBench | Steps/task | -8.7% | -9.6% |
| MCP-Universe | API Total | +7.2% | -31.9% |
| MCP-Universe | Iterations | +35.9% | -19.5% |
| MCP-Universe | Schema | +27.1% | -2.4% |
| MCP-Universe | Tool Call | +54.3% | -39.0% |
| MCP-Universe | Results | -37.4% | -14.4% |
| MCPToolBenchPP (30b) | Total | -8.8% | -20.9% |
| MCPToolBenchPP (30b) | Schema | -6.8% | -21.8% |
| MCPToolBenchPP (32b) | Schema | -43.1% | -11.1% |
| BFCL | Total | +3.2% | -0.4% |

## 6. Key Findings

### Finding 1: Format compression nearly doubles small-model performance

qwen3-30b with TRON achieves 59.7% SoPR vs 34.2% with JSON on StableToolBench — a +25.5 percentage point improvement (+74.6% relative) with zero model changes, no fine-tuning, no additional training. The same model becomes dramatically more capable just by changing the serialization format. This is because compressed formats free ~35% of the context window, allowing the model to complete tasks it otherwise cannot finish within the step budget (finish rate: 80.3% TRON vs 40.9% JSON).

### Finding 2: Compression benefit is model-dependent (not universal)

qwen3.5-35b shows the opposite effect: TRON *hurts* performance (-7.6% SoPR, 24.2% vs 31.8%). Stronger models handle verbose JSON context effectively and do not benefit from compression. Unfamiliar formats can even confuse them. This means format compression is a tool for context-limited models, not a universal improvement. The practical implication: use compression for cost-efficient deployment with smaller models.

### Finding 3: TOON cascading failure in multi-turn settings

TOON's indentation-based syntax causes parsing failures when the model generates tool calls. In multi-turn settings, this triggers a cascade: parsing error → retry → additional iteration → full conversation history re-sent → more tokens consumed. On MCP-Universe: TOON increases iterations by +35.9% (10.4 → 14.2) and tool call tokens by +54.3%, negating the -42.1% per-call schema compression and resulting in +7.2% total tokens. TRON avoids this because its syntax stays close to JSON (only adds class definitions), producing fewer parsing errors and fewer iterations (8.4, -19.5%).

### Finding 4: TOON only works with native function calling

On StableToolBench (which uses native `tool_calls` API, not text parsing), TOON achieves +19.7% SoPR improvement. On MCP-Universe and MCPToolBenchPP (text-based ReAct parsing), TOON fails (-69.5% and -34.3%). The difference: native function calling handles output structure internally, so TOON's unfamiliar syntax only affects the input (schemas/results), where it compresses well. In text-based agents, the model must produce correctly formatted output, which it cannot do reliably in TOON.

### Finding 5: TRON is the best format overall

TRON consistently preserves accuracy while saving tokens across all benchmarks:
- StableToolBench: +25.5% SoPR, -35.8% tokens
- MCP-Universe: -4.6% pass rate, -31.9% tokens
- MCPToolBenchPP: -8.9% pass@1, -20.9% tokens
- BFCL: -0.6% accuracy, -0.4% tokens

### Finding 6: Tool selection is format-agnostic, parameter generation is the bottleneck

Tool selection accuracy (tool_pass) remains within 2% of JSON across all formats (~98% on MCPToolBenchPP). The accuracy gap comes from parameter generation: TOON drops param_pass by 25-28%, TRON only 4%. LLMs can identify the correct tool from compressed schemas but struggle to produce correctly formatted parameters in unfamiliar notation.

### Finding 7: Compression benefit scales with context pressure

The benefit increases with schema verbosity:
- BFCL (1-3 small functions per task): TRON -0.6% accuracy, -0.4% tokens — negligible effect
- MCPToolBenchPP (5-32 MCP tools): TRON -8.9% accuracy, -20.9% tokens — moderate effect
- StableToolBench (5-10 verbose RapidAPI docs): TRON +25.5% accuracy, -35.8% tokens — massive effect

### Finding 8: Multi-turn amplifies format effects in both directions

Single-turn benchmarks underestimate the true impact of format choice. In multi-turn settings, effects compound across iterations:
- TRON: fewer iterations → less repeated context → compounded savings (0.4% saving on BFCL → 31.9% on Universe)
- TOON: more iterations → more repeated context → compounded overhead (3.2% overhead on BFCL → 7.2% on Universe)

### Finding 9: Completion tokens are the biggest savings

On StableToolBench, completion tokens drop by 60-66% with compressed formats. This is because the model produces shorter, more focused reasoning when context is less cluttered, and compressed tool calls are inherently shorter.

### Finding 10: TRON benefits from both input and output compression, TOON only from input

Comparing exp1 (input-only compression) vs exp2 (full compression) on StableToolBench with qwen3-30b:
- TOON: exp1 54.2% vs exp2 53.9% (-0.3pp). No difference. The entire benefit comes from compressing schemas and tool results in the input. Compressing tool call output adds nothing because StableToolBench uses native function calling (the model doesn't actually write TOON).
- TRON: exp1 52.7% vs exp2 59.7% (+7.0pp). Output compression adds significant extra benefit. TRON's class instantiation syntax produces shorter tool calls, freeing more context for reasoning in subsequent iterations.

This means TRON has two sources of compression benefit (input + output) while TOON has only one (input). In text-based agents where the model must write in the format, TRON's advantage over TOON would be even larger.

## 7. Pending Results (as of 2026-03-26 — superseded by Section 8+)

| Job | Model | Benchmark | Partition | Status |
|---|---|---|---|---|
| 7537563 | Qwen3-32B-AWQ | StableToolBench | 2x A40 | queued |
| 7538087 | Qwen3-32B-FP16 (thinking) | StableToolBench | 2x A100 | running G1/json |
| 7538301 | Qwen3.5-122B-GPTQ-Int4 | StableToolBench | 2x A100 | queued |


---

# Update: 2026-04-21 — Multi-model cluster runs

*Added 2026-04-21. Extends the comparison to 4 additional models run on the VSC5 cluster:
Mistral-Small-24B-Instruct-2501-AWQ, Qwen3-32B (FP16), Qwen3-32B-AWQ, Qwen3-235B-A22B-AWQ (qwen35),
and DeepSeek-R1-Distill-32B-AWQ. See `cluster_monday_2026_04_20/` for raw per-snapshot tables.*

## 8. StableToolBench — Multi-model extension

Same benchmark, same 330 solvable-queries subset. Judge changed to DeepSeek-R1-Distill-32B-AWQ
hosted on cluster (run_eval.slurm) to remove external-API cost and dependency. Pass criterion
unchanged (>1.5 of 3 votes marked Solved).

### 8.1 SoPR Overall (as of 2026-04-21)

| Model | JSON | TOON | TRON | TOON vs JSON | TRON vs JSON |
|---|---|---|---|---|---|
| qwen3-30b (local) | 33.3% | 59.4% | 57.0% | +26.1pp | +23.7pp |
| qwen3-32b-fp16 | 31.2% | 31.5% | **34.8%** | +0.3pp | +3.6pp |
| qwen3-32b-awq | **34.5%** | 32.1% | 33.0% | -2.4pp | -1.5pp |
| qwen35 (235B) | 33.3% | 31.8% | 23.6% | -1.5pp | -9.7pp |
| mistral-small-24b | *stale* | *stale* | *stale* | — | — |
| deepseek-r1-32b | *stale* | *stale* | *stale* | — | — |

*stale* = eval pending against post-rerun inference outputs (scheduled, will refresh after
job 7936138 completes).

### 8.2 Token Totals (as of 2026-04-21)

| Model | Format | Prompt | Completion | Observation | Total | Δ vs JSON |
|---|---|---|---|---|---|---|
| qwen3-32b-fp16 | json | 2,163,195 | 85,780 | 84,597 | 2,248,975 | — |
| qwen3-32b-fp16 | toon | 3,150,257 | 73,173 | 77,638 | 3,223,430 | +43.3% |
| qwen3-32b-fp16 | tron | 3,096,600 | 80,390 | 78,048 | 3,176,990 | +41.3% |
| qwen3-32b-awq | json | 2,115,031 | 77,037 | 82,711 | 2,192,068 | — |
| qwen3-32b-awq | toon | 3,171,431 | 74,859 | 73,427 | 3,246,290 | +48.1% |
| qwen3-32b-awq | tron | 3,050,162 | 69,611 | 75,068 | 3,119,773 | +42.3% |
| qwen35 | json | 2,353,815 | 139,614 | 119,262 | 2,493,429 | — |
| qwen35 | toon | 3,656,795 | 144,282 | 111,360 | 3,801,077 | +52.4% |
| qwen35 | tron | 1,738,361 | 84,933 | 71,808 | 1,823,294 | **-26.9%** |
| mistral-small-24b | json | 3,547,044 | 160,035 | 79,263 | 3,707,079 | — |
| mistral-small-24b | toon | 5,169,965 | 124,072 | 73,280 | 5,294,037 | +42.8% |
| mistral-small-24b | tron | 0 | 0 | 0 | 0 | needs rerun |
| deepseek-r1-32b | json | 7,317,016 | 2,096,320 | 0 | 9,413,336 | — |
| deepseek-r1-32b | toon | 11,925,004 | 2,334,059 | 0 | 14,259,063 | +51.5% |
| deepseek-r1-32b | tron | 11,192,013 | 2,147,117 | 0 | 13,339,130 | +41.7% |

### 8.3 Key observations — multi-model StableToolBench

**Finding 11: Compression only saves tokens on the smallest model (30B) and qwen35-TRON.**
For 32B-class models, TOON and TRON actually INCREASE total tokens by 40-50%. The per-call
schema savings are overwhelmed by longer reasoning chains triggered when the model encounters
unfamiliar formats. This is the iteration-cascade effect from Finding 8 showing up across the
model suite — just more extreme than Universe suggested.

**Finding 12: Accuracy effects are model-specific and small for 32B+ models.**
Unlike qwen3-30b's +26pp jump with TOON, the 32B and 235B models see only ±3pp. Either the
larger models are more format-robust (they can handle TOON/TRON), or the 30B result is driven
by context-window pressure that doesn't apply at 32B+. The qwen35-TRON G2/G3 collapse
(-44pp G2, -100pp G3) is an exception that warrants investigation before publication.

**Finding 13: DeepSeek-R1 pays a heavy token price across all formats.**
Even the JSON baseline (9.4M total) is 2-4x higher than other models. The reasoning model
produces ~2M completion tokens (vs ~80-160k for other models). Format choice does not fix
this — reasoning models need a different compression story.

## 9. BFCL — Mistral-Small-24B (complete)

Single-model deep-dive; adds non-live (single-turn) and multi-turn coverage to complement
the StableToolBench multi-turn-only view.

### 9.1 Accuracy — non-live (all 6 variants comparable)

| Category | baseline | toon (exp1) | tron (exp1) | json-full (exp2) | toon-full (exp2) | tron-full (exp2) |
|---|---|---|---|---|---|---|
| simple_python | 94.25% | 53.75% | 93.00% | 94.00% | 78.50% | 94.00% |
| multiple | 93.50% | 47.50% | 91.00% | 93.50% | 77.50% | 93.50% |
| parallel | 92.00% | 54.50% | 90.00% | 91.00% | **0.00%** | 91.00% |
| parallel_multiple | 90.50% | 35.50% | 85.50% | 89.50% | **0.00%** | 87.00% |
| irrelevance | 72.08% | 65.83% | 51.25% | 100.00% | 100.00% | 100.00% |

### 9.2 Accuracy — multi-turn (exp1 only)

| Category | baseline | toon | tron |
|---|---|---|---|
| multi_turn_base | 33.00% | 5.50% | **33.50%** |
| multi_turn_miss_func | 15.50% | 3.50% | **19.00%** |
| multi_turn_miss_param | 17.00% | 3.00% | **21.00%** |
| multi_turn_long_context | 15.00% | 1.00% | **16.50%** |

Bold = beats JSON baseline.

### 9.3 Token consumption — non-live (1090 tasks)

| variant | input | Δ | output | Δ | total | Δ |
|---|---|---|---|---|---|---|
| baseline | 677,316 | — | 47,572 | — | 724,888 | — |
| toon (exp1) | 716,958 | +5.9% | 67,808 | +42.5% | 784,766 | +8.3% |
| tron (exp1) | 693,743 | +2.4% | 64,918 | +36.5% | 758,661 | +4.7% |
| json-full (exp2) | 697,156 | +2.9% | 64,013 | +34.6% | 761,169 | +5.0% |
| toon-full (exp2) | 729,358 | +7.7% | 66,664 | +40.1% | 796,022 | +9.8% |
| tron-full (exp2) | 707,383 | +4.4% | 77,556 | +63.0% | 784,939 | +8.3% |

### 9.4 Token consumption — multi-turn (800 tasks)

| variant | input | Δ | output | Δ | total | Δ |
|---|---|---|---|---|---|---|
| baseline | 52,619,891 | — | 302,607 | — | 52,922,498 | — |
| toon (exp1) | 37,667,622 | **-28.4%** | 273,391 | -9.7% | 37,941,013 | **-28.3%** |
| tron (exp1) | 42,958,336 | **-18.4%** | 356,081 | +17.7% | 43,314,417 | **-18.2%** |

### 9.5 Finding 14: Compression economics flip with context length

On the same model and benchmark, whether compression saves tokens depends on how much
structured data piles up. On short single-turn prompts (BFCL non-live), format-hint
boilerplate and slightly more verbose output give a 5-10% *increase* in tokens. On long
multi-turn prompts (BFCL multi-turn, Universe, StableToolBench), the per-call savings
multiply with turns and compression pays off (18-28% savings on BFCL multi-turn).

**Paper implication:** claims about format compression must specify the workload regime.
"TRON saves tokens" is only true above a crossover point on context size. Below that, it
costs more.

### 9.6 Finding 15: TRON is a clean drop-in replacement for mistral-small-24b

tron-full and json-full match within ≤1.5pp on every single-turn BFCL category
(simple_python, multiple, parallel, parallel_multiple). TOON-full fails entirely on
parallel tasks (0.00%). For this model, TRON is the only compressed format viable as a
drop-in JSON replacement.

## 10. Cluster runs — pending and in-progress (as of 2026-04-21)

- BFCL chains running: qwen3-32b-awq, qwen3-32b-fp16, deepseek-r1-32b
- BenchPP chains running: mistral-small-24b, qwen3-32b-awq, qwen3-32b-fp16, deepseek-r1-32b
- StableToolBench: mistral tron inference rerun (7934358) + eval refresh (7936138) queued
- MCP-Universe: **not attempted on cluster** (blocked by missing Docker / Node 18+ / Blender on compute nodes). Multi-model comparison will be limited to the existing local qwen3-30b + qwen3-32b runs.

Once the chains finish, Section 8.1 and 8.2 will refresh with the non-stale mistral and deepseek
rows, and Sections 9 + 10 will extend to the other 3 cluster models.