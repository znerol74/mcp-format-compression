# BFCL Token Consumption — Mistral-Small-24B-Instruct-2501-AWQ

Source: `gorilla/berkeley-function-call-leaderboard/result/mistral-small-24b*/`
Per-task `input_token_count` and `output_token_count` fields from BFCL result JSONL, summed across tasks.

## Non-live (single-turn, 1090 tasks)

All 6 variants comparable; schemas are short and appear once per task.

| variant | input tokens | Δ vs baseline | output tokens | Δ vs baseline | total | Δ vs baseline |
|---|---|---|---|---|---|---|
| baseline | 677,316 | +0.0% | 47,572 | +0.0% | 724,888 | +0.0% |
| toon (exp1) | 716,958 | +5.9% | 67,808 | +42.5% | 784,766 | +8.3% |
| tron (exp1) | 693,743 | +2.4% | 64,918 | +36.5% | 758,661 | +4.7% |
| json-full (exp2) | 697,156 | +2.9% | 64,013 | +34.6% | 761,169 | +5.0% |
| toon-full (exp2) | 729,358 | +7.7% | 66,664 | +40.1% | 796,022 | +9.8% |
| tron-full (exp2) | 707,383 | +4.4% | 77,556 | +63.0% | 784,939 | +8.3% |

**Finding:** all compressed variants use MORE tokens than the baseline on single-turn BFCL. Input tokens grow ~3-8% due to format-hint boilerplate; output tokens grow 35-63% because compressed output formats are all more verbose than Python's `[func_name(arg=val)]` notation.

## Multi-turn (800 tasks, exp1 variants only)

Long accumulated context — schemas + tool results pile up across turns.

| variant | input tokens | Δ vs baseline | output tokens | Δ vs baseline | total | Δ vs baseline |
|---|---|---|---|---|---|---|
| baseline | 52,619,891 | +0.0% | 302,607 | +0.0% | 52,922,498 | +0.0% |
| toon (exp1) | 37,667,622 | **-28.4%** | 273,391 | -9.7% | 37,941,013 | **-28.3%** |
| tron (exp1) | 42,958,336 | **-18.4%** | 356,081 | +17.7% | 43,314,417 | **-18.2%** |

**Finding:** multi-turn is where compression pays off. TOON saves 28%, TRON saves 18% total tokens.

## Cross-finding: economics flip with context length

- **Non-live (short prompts)** — compression format-hint overhead exceeds savings. TRON and TOON both lose.
- **Multi-turn (long accumulated context)** — compression savings dominate. Both TRON and TOON save tokens.

Pair with accuracy:

| | non-live accuracy | non-live tokens | multi-turn accuracy | multi-turn tokens |
|---|---|---|---|---|
| TRON vs JSON | ~equal | worse (+5%) | ~equal or better | better (-18%) |
| TOON vs JSON | much worse (-30 to -40pp) | worse (+8%) | much worse (-30pp) | better (-28%) |

**For mistral-small-24b, the paper claim is:** TRON is a clean Pareto improvement over JSON in multi-turn workloads, and neutral (slightly wasteful) in single-turn. TOON is strictly worse than JSON in both scenarios for this model.
