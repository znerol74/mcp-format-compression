# Cluster Results Snapshot — 2026-04-20

Snapshot taken Monday 2026-04-20 after the weekend cluster run.
Cluster chain jobs (BFCL + BenchPP × remaining models) still in progress at time of snapshot.

## Scope of this snapshot

| Benchmark | Complete | Partial | Missing / Stale |
|---|---|---|---|
| BFCL | mistral-small-24b (all 6 variants × 9 categories) | deepseek-r1-32b (baseline + partial toon), qwen3-32b-awq (baseline only), qwen3-32b-fp16 (baseline only) | — |
| StableToolBench | 6 models × 3 formats (inference) + eval | mistral + deepseek EVAL stale — need re-eval on cluster against the new Apr 16 / Apr 19 inference | — |
| MCPToolBenchPP | — | mistral json+toon salvaged; qwen3-32b-awq + deepseek-r1-32b baseline only | qwen3-32b-fp16 (failed vLLM, chain resubmit in progress) |
| MCP-Universe | — | — | All models (ran only mistral-small-24b smoke tests; cluster fanout abandoned due to MCP server dependency blockers — see notes). |

Pipelines validated end-to-end for BFCL + BenchPP (judge dispatch fixed).

## Known-correct numbers (this snapshot)

- **BFCL mistral-small-24b** — see `bfcl_mistral_accuracy.md` and `bfcl_mistral_tokens.md`.
- **StableToolBench SoPR** (all 6 models, 3 formats) — see `stabletoolbench_sopr.md`. **Note: mistral and deepseek numbers on that page reflect OLD eval; those need to be re-evaluated on cluster against the Apr 16 / Apr 19 reruns.**

## Files in this directory

- `README.md` — this file
- `bfcl_mistral_accuracy.md` — BFCL mistral accuracy table (complete, 6 variants × 9 categories)
- `bfcl_mistral_tokens.md` — BFCL mistral token-consumption table + savings calculations
- `stabletoolbench_sopr.md` — SoPR per model × group × format
- `stabletoolbench_tokens.md` — StableToolBench token consumption per tag

## Re-evaluation TODO

- [ ] Re-run `run_eval.slurm` on cluster to refresh mistral + deepseek StableToolBench eval against the new inference outputs
- [ ] Salvage weekend BenchPP partial logs via `cluster/salvage_benchpp_logs.py` (already done on cluster)
- [ ] BenchPP `finance` MCP server returns HTTP 500 — investigate whether upstream API dependency is missing on cluster
- [ ] BenchPP judge (Qwen3-32B-AWQ) produces JSON-parse errors — param_pass@1 numbers will be unreliable until judge is swapped or output format tightened

## MCP-Universe note

Cluster cannot run full MCP-Universe coverage — three of the five local categories require OS-level
dependencies unavailable on compute nodes:

| Category | Blocker |
|---|---|
| location_navigation | `npx` / Node.js 18+ not on cluster |
| browser_automation | Node + Playwright chromium binary |
| repository_management | Docker daemon |
| 3d_design | Running Blender GUI (Blender addon needs localhost:9876) |

Only financial_analysis and web_search are cluster-viable. Decision: keep MCP-Universe
results from local qwen3-30b and qwen3-32b runs only, do not attempt cluster fanout.
