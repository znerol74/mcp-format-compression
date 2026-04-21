# StableToolBench SoPR — All Cluster Models

Source: `StableToolBench/eval_results/<tag>/<group>_instruction_results.json`
Judge: DeepSeek-R1-32B-AWQ on cluster (run_eval.slurm)
Pass criterion: > 1.5 of 3 evaluator votes marked "Solved".

**STALENESS WARNING:** mistral and deepseek rows below reflect the OLD weekend-before eval.
Their inference was re-run (mistral 2026-04-16, deepseek 2026-04-19). The eval must be re-run on
cluster against the new inference outputs before these numbers are correct — they currently
read 0% because the old eval saw empty stub result files.

| Model | Fmt | G1 | G2 | G3 | Overall |
|---|---|---|---|---|---|
| **local** (qwen3:30b-instruct) | json | 50/163 (30.7%) | 42/106 (39.6%) | 18/61 (29.5%) | **110/330 (33.3%)** |
| local | toon | 102/163 (62.6%) | 62/106 (58.5%) | 32/61 (52.5%) | **196/330 (59.4%)** |
| local | tron | 97/163 (59.5%) | 58/106 (54.7%) | 33/61 (54.1%) | **188/330 (57.0%)** |
| **qwen3-32b-fp16** | json | 69/163 (42.3%) | 30/106 (28.3%) | 4/61 (6.6%) | 103/330 (31.2%) |
| qwen3-32b-fp16 | toon | 68/163 (41.7%) | 33/106 (31.1%) | 3/61 (4.9%) | 104/330 (31.5%) |
| qwen3-32b-fp16 | tron | 71/163 (43.6%) | 38/106 (35.8%) | 6/61 (9.8%) | **115/330 (34.8%)** |
| **qwen3-32b-awq** | json | 75/163 (46.0%) | 33/106 (31.1%) | 6/61 (9.8%) | **114/330 (34.5%)** |
| qwen3-32b-awq | toon | 68/163 (41.7%) | 33/106 (31.1%) | 5/61 (8.2%) | 106/330 (32.1%) |
| qwen3-32b-awq | tron | 66/163 (40.5%) | 37/106 (34.9%) | 6/61 (9.8%) | 109/330 (33.0%) |
| **qwen35** (Qwen3-235B-A22B-AWQ) | json | 67/163 (41.1%) | 29/106 (27.4%) | 14/61 (23.0%) | **110/330 (33.3%)** |
| qwen35 | toon | 62/163 (38.0%) | 29/106 (27.4%) | 14/61 (23.0%) | 105/330 (31.8%) |
| qwen35 | tron | 68/163 (41.7%) | 10/106 (9.4%) | 0/61 (0.0%) | 78/330 (23.6%) |
| **mistral-small-24b** (STALE) | json | 0% | 0% | 0% | 0% |
| mistral-small-24b (STALE) | toon | 0% | 0% | 0% | 0% |
| mistral-small-24b (STALE) | tron | 0% | 0% | 0% | 0% |
| **deepseek-r1-32b** (STALE) | json | 0% | 0% | 0% | 0% |
| deepseek-r1-32b (STALE) | toon | 0% | 0% | 0% | 0% |
| deepseek-r1-32b (STALE) | tron | 0% | 0% | 0% | 0% |

**Bold** marks the best format per model on Overall SoPR.

## Observations (non-stale rows only)

- **local (qwen3:30b-instruct): compression helps massively.** TOON +26pp over JSON, TRON +24pp. Consistent with the hypothesis that small models benefit most from reduced context pressure.
- **qwen3-32b-fp16: TRON best (+3.6pp over JSON), TOON essentially tied with JSON.** Much smaller delta than the 30B local model.
- **qwen3-32b-awq: JSON wins (34.5%), TOON/TRON both slightly worse.** Quantization and format interact unexpectedly; AWQ + format-compressed schemas degrades slightly.
- **qwen35 (235B MoE): TRON G2/G3 collapse** (9.4%, 0.0%). Possibly the MoE routing or quantization interacts badly with TRON's class-definition syntax. Worth investigating before publication.

## Actions needed

1. Re-run cluster `run_eval.slurm` on the new mistral and deepseek inference outputs
2. Resync `eval_results_local/` down to laptop
3. Regenerate this table
