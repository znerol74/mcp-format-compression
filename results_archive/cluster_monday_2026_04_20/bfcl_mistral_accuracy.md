# BFCL Accuracy — Mistral-Small-24B-Instruct-2501-AWQ

Source: `gorilla/berkeley-function-call-leaderboard/score/mistral-small-24b*/`
Cluster job: `bfcl-mistral` (7855396), completed 2026-04-17T21:12:46 after 11h46m.
Deterministic AST matching; no LLM judge.

Variant naming convention:
- `baseline`    — JSON tool schemas + Python tool-call output (exp1 JSON baseline)
- `toon`        — TOON tool schemas + Python tool-call output (exp1 TOON)
- `tron`        — TRON tool schemas + Python tool-call output (exp1 TRON)
- `json-full`   — JSON schemas + JSON tool-call output (exp2 baseline)
- `toon-full`   — TOON schemas + TOON tool-call output (exp2 TOON)
- `tron-full`   — TRON schemas + TRON tool-call output (exp2 TRON)

## Non-live (single-turn) — all 6 variants comparable

| Category | baseline | toon (exp1) | tron (exp1) | json-full (exp2) | toon-full (exp2) | tron-full (exp2) |
|---|---|---|---|---|---|---|
| simple_python | 94.25% | 53.75% | 93.00% | 94.00% | 78.50% | 94.00% |
| multiple | 93.50% | 47.50% | 91.00% | 93.50% | 77.50% | 93.50% |
| parallel | 92.00% | 54.50% | 90.00% | 91.00% | **0.00%** | 91.00% |
| parallel_multiple | 90.50% | 35.50% | 85.50% | 89.50% | **0.00%** | 87.00% |
| irrelevance | 72.08% | 65.83% | 51.25% | 100.00% | 100.00% | 100.00% |

## Multi-turn — exp1 variants only (exp2 not compatible with multi-turn BFCL evaluator)

| Category | baseline | toon (exp1) | tron (exp1) |
|---|---|---|---|
| multi_turn_base | 33.00% | 5.50% | **33.50%** |
| multi_turn_miss_func | 15.50% | 3.50% | **19.00%** |
| multi_turn_miss_param | 17.00% | 3.00% | **21.00%** |
| multi_turn_long_context | 15.00% | 1.00% | **16.50%** |

**Bold** = beats JSON baseline.

## Observations

1. **tron-full ≈ json-full ≈ baseline** on simple/multiple/parallel — TRON is a clean drop-in with essentially zero accuracy cost.
2. **toon-full catastrophically fails on parallel tasks (0.00%)** while keeping reasonable simple/multiple scores. The TOON output format breaks when multiple calls must be emitted in parallel.
3. **toon-exp1 (Python output) drops 30-40 points** everywhere — TOON as INPUT format alone already hurts this model.
4. **TRON slightly beats baseline on all 4 multi-turn categories**. Smallest gain on multi_turn_base (+0.5), largest on multi_turn_miss_param (+4.0).
5. **Irrelevance jumps to 100% in all exp2 variants**. Likely an evaluator artifact: JSON/TOON/TRON tool-call output cleanly expresses "no tool needed"; Python output gets marked wrong on the same task.
