# StableToolBench Token Consumption — All Cluster Tags

Source: per-task `answer_generation.prompt_tokens` / `completion_tokens` / `observation_tokens` / `total_tokens`
in `StableToolBench/results/<tag>/*/<task>.json`, summed across all tasks.

`n` is the number of task files with the tag. 330 = complete run (163 G1 + 106 G2 + 61 G3).

| tag | n | prompt | completion | observation | total |
|---|---|---|---|---|---|
| local_json_tcjson | 330 | 4,773,429 | 311,278 | 68,043 | 5,084,707 |
| local_toon_tcjson | 330 | 3,239,820 | 117,639 | 97,629 | 3,357,459 |
| local_toon_tctoon | 330 | 3,154,615 | 107,182 | 90,594 | 3,261,797 |
| local_tron_tcjson | 330 | 3,229,086 | 126,380 | 95,828 | 3,355,466 |
| local_tron_tctron | 330 | 3,142,789 | 123,679 | 84,627 | 3,266,468 |
| qwen3-32b-fp16_json_tcjson | 330 | 2,163,195 | 85,780 | 84,597 | 2,248,975 |
| qwen3-32b-fp16_toon_tctoon | 330 | 3,150,257 | 73,173 | 77,638 | 3,223,430 |
| qwen3-32b-fp16_tron_tctron | 330 | 3,096,600 | 80,390 | 78,048 | 3,176,990 |
| qwen3-32b-fp16-think_json_tcjson | 330 | 2,208,341 | 464,074 | 76,973 | 2,672,415 |
| qwen3-32b-fp16-think_toon_tctoon | 213 | 2,004,580 | 289,502 | 54,692 | 2,294,082 |
| qwen3-32b-awq_json_tcjson | 330 | 2,115,031 | 77,037 | 82,711 | 2,192,068 |
| qwen3-32b-awq_toon_tctoon | 330 | 3,171,431 | 74,859 | 73,427 | 3,246,290 |
| qwen3-32b-awq_tron_tctron | 330 | 3,050,162 | 69,611 | 75,068 | 3,119,773 |
| qwen35_json_tcjson | 330 | 2,353,815 | 139,614 | 119,262 | 2,493,429 |
| qwen35_toon_tctoon | 330 | 3,656,795 | 144,282 | 111,360 | 3,801,077 |
| qwen35_tron_tctron | 330 | 1,738,361 | 84,933 | 71,808 | 1,823,294 |
| mistral-small-24b_json_tcjson | 330 | 3,547,044 | 160,035 | 79,263 | 3,707,079 |
| mistral-small-24b_toon_tctoon | 330 | 5,169,965 | 124,072 | 73,280 | 5,294,037 |
| mistral-small-24b_tron_tctron | 330 | 0 | 0 | 0 | 0 |
| deepseek-r1-32b_json_tcjson | 330 | 7,317,016 | 2,096,320 | 0 | 9,413,336 |
| deepseek-r1-32b_toon_tctoon | 330 | 11,925,004 | 2,334,059 | 0 | 14,259,063 |
| deepseek-r1-32b_tron_tctron | 330 | 11,192,013 | 2,147,117 | 0 | 13,339,130 |

## Anomalies to investigate

1. **mistral-small-24b_tron_tctron = 0 across the board.** Inference result files exist (n=330) but all have 0 tokens — indicating the tron_tctron run never actually produced LLM output. This tag needs to be rerun on cluster.
2. **qwen3-32b-fp16-think_toon_tctoon** only has n=213, not 330. Missing 117 tasks — presumably G3 incomplete. Low priority if we're not using the "-think" variant for the main comparison.
3. **deepseek-r1-32b observation=0** across all three formats — but prompt/completion non-zero. Suggests the ReACT loop is not reporting tool-observation tokens (may be a pipeline instrumentation gap specific to the NO_THINK=1 variant).
4. **deepseek TOON + TRON use MORE tokens than JSON** (14.3M vs 9.4M, 13.3M vs 9.4M). Very unusual — typically compression saves tokens. Either the reasoning model takes longer paths with unfamiliar formats, or there's an issue with the tokenization of compressed schemas for this model.

## Per-model compression % (total, complete runs only)

| Model | JSON baseline | TOON total | TOON Δ | TRON total | TRON Δ |
|---|---|---|---|---|---|
| local | 5,084,707 | 3,261,797 | **-35.9%** | 3,266,468 | **-35.8%** |
| qwen3-32b-fp16 | 2,248,975 | 3,223,430 | +43.3% | 3,176,990 | +41.3% |
| qwen3-32b-awq | 2,192,068 | 3,246,290 | +48.1% | 3,119,773 | +42.3% |
| qwen35 | 2,493,429 | 3,801,077 | +52.4% | 1,823,294 | **-26.9%** |
| mistral-small-24b | 3,707,079 | 5,294,037 | +42.8% | (no data) | — |
| deepseek-r1-32b | 9,413,336 | 14,259,063 | +51.5% | 13,339,130 | +41.7% |

**Striking pattern**: only the local 30B model and qwen35-TRON achieve compression savings. For 32B-class models, TOON/TRON INCREASES total tokens — the compressed format triggers longer reasoning chains or more iterations, overwhelming the per-call savings.

This is the multi-turn iteration cascade effect from the paper (Section 5.3 in the current draft) showing up across the model suite.
