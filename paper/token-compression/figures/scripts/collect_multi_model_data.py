#!/usr/bin/env python3
"""Collect per-(model, format, benchmark) metrics from all four benchmarks.

Produces multi_model_data.json with the shape:
{
  "<model_tag>": {
    "TOON": {"<benchmark>": {"tok_pct": ..., "acc_pct": ..., "acc_abs": ..., "acc_json": ...}},
    "TRON": {...}
  },
  ...
}
where tok_pct / acc_pct are computed vs the JSON baseline for the SAME model on the SAME benchmark.
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_PATH = Path(__file__).resolve().parent / "multi_model_data.json"

# ─── Model name normalization ──────────────────────────────────────────────
# Maps raw strings from each benchmark into canonical labels used in the paper.
MODEL_ALIAS = {
    # BenchPP
    "qwen3-32b": "Qwen3-32B",
    "qwen3-30b-instruct-openwebui": "Qwen3-30B-Instruct",
    "qwen3:30b-instruct-openwebui": "Qwen3-30B-Instruct",
    # Universe
    "qwen3-30b-instruct": "Qwen3-30B-Instruct",
    "qwen3:30b-instruct": "Qwen3-30B-Instruct",
    # STB tags
    "local": "Qwen3-30B-Instruct",
    "qwen35": "Qwen3.5-Next",
    "qwen3-32b-awq": "Qwen3-32B-AWQ",
    "qwen3-32b-fp16": "Qwen3-32B-FP16",
    "qwen3-32b-fp16-think": "Qwen3-32B-FP16 (think)",
    "deepseek-r1-32b": "DeepSeek-R1-32B",
    "mistral-small-24b": "Mistral-Small-24B",
}


def canon_model(raw):
    return MODEL_ALIAS.get(raw, raw)


# ─── MCPToolBenchPP ────────────────────────────────────────────────────────
def collect_benchpp():
    """Returns {model_canon: {fmt: {category: {tokens, pass}}}}."""
    log_dir = ROOT / "MCPToolBenchPP" / "logs"
    # (category, fmt, model) -> (file_path, timestamp)
    latest = {}
    for f in log_dir.rglob("*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
        except Exception:
            continue
        info = data.get("run_info", {})
        metrics = data.get("metrics", [])
        if not metrics:
            continue
        m = metrics[0]
        num_tasks = m.get("num_tasks", 0)
        if num_tasks < 10:
            continue
        model = info.get("model", "?")
        fmt = info.get("tool_format", "json")
        cat = info.get("category", "?")
        # Use file mtime as tiebreaker for latest
        ts = f.stat().st_mtime
        key = (cat, fmt, model)
        if key not in latest or latest[key][1] < ts:
            latest[key] = (f, ts, data)

    by_model_fmt = defaultdict(lambda: defaultdict(dict))
    for (cat, fmt, model), (_f, _ts, data) in latest.items():
        info = data["run_info"]
        m = data["metrics"][0]
        tu = info.get("token_usage", {})
        mkey = canon_model(model)
        by_model_fmt[mkey][fmt][cat] = {
            "num_tasks": m.get("num_tasks", 0),
            "pass_at_1": m.get("pass@1", 0),
            "tokens": tu.get("total_api_total_tokens", 0),
        }
    return by_model_fmt


# ─── MCP-Universe ──────────────────────────────────────────────────────────
def parse_universe_report(path):
    with open(path) as f:
        content = f.read()
    tasks = []
    for line in content.split("\n"):
        if not line.startswith("|**"):
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 13:
            continue
        try:
            passed = int(parts[1])
            not_passed = int(parts[2])
            total_tok = int(parts[8])
            tasks.append({"passed": passed, "not_passed": not_passed, "total_tok": total_tok})
        except (ValueError, IndexError):
            continue
    if not tasks:
        return None
    total = len(tasks)
    tp = sum(1 for t in tasks if t["not_passed"] == 0 and t["passed"] > 0)
    return {
        "num_tasks": total,
        "pass_rate": tp / total if total else 0,
        "tokens": sum(t["total_tok"] for t in tasks),
    }


def collect_universe():
    report_dir = ROOT / "MCP-Universe" / "log" / "test_full_report"
    by_model_fmt = defaultdict(lambda: defaultdict(dict))
    if not report_dir.exists():
        return by_model_fmt
    for f in report_dir.iterdir():
        if not f.is_file():
            continue
        name = f.name
        m = re.match(r"^(?P<cat>.+?)_(?P<fmt>json|toon|tron)_tc(?P<tc>json|toon|tron)_(?P<model>.+)$", name)
        if not m:
            continue
        cat = m.group("cat")
        fmt = m.group("fmt")
        tc = m.group("tc")
        model = m.group("model")
        # Only take the "full compression" variant: fmt == tc
        if fmt != tc:
            continue
        parsed = parse_universe_report(f)
        if not parsed:
            continue
        mkey = canon_model(model)
        parsed["category"] = cat
        by_model_fmt[mkey][fmt][cat] = parsed
    return by_model_fmt


# ─── BFCL ──────────────────────────────────────────────────────────────────
BFCL_MODEL_MAP = {
    # (registry_dir, accuracy_row_contains) -> (canon_model, format, scope)
    "qwen3-32b-openrouter":          ("Qwen3-32B", "json", "input"),
    "qwen3-32b-openrouter-toon":     ("Qwen3-32B", "toon", "input"),
    "qwen3-32b-openrouter-tron":     ("Qwen3-32B", "tron", "input"),
    "qwen3-32b-openrouter-json-full":("Qwen3-32B", "json", "full"),
    "qwen3-32b-openrouter-toon-full":("Qwen3-32B", "toon", "full"),
    "qwen3-32b-openrouter-tron-full":("Qwen3-32B", "tron", "full"),
    "qwen3-30b-local":               ("Qwen3-30B-Instruct", "json", "input"),
    "qwen3-30b-local-toon":          ("Qwen3-30B-Instruct", "toon", "input"),
    "qwen3-30b-local-tron":          ("Qwen3-30B-Instruct", "tron", "input"),
    "qwen3-30b-local-json-full":     ("Qwen3-30B-Instruct", "json", "full"),
    "qwen3-30b-local-toon-full":     ("Qwen3-30B-Instruct", "toon", "full"),
    "qwen3-30b-local-tron-full":     ("Qwen3-30B-Instruct", "tron", "full"),
}


def collect_bfcl():
    """Use Overall Acc from data_overall.csv; sum tokens from per-task JSONL result files."""
    score_dir = ROOT / "gorilla" / "berkeley-function-call-leaderboard" / "score"
    result_dir = ROOT / "gorilla" / "berkeley-function-call-leaderboard" / "result"
    overall = score_dir / "data_overall.csv"
    if not overall.exists():
        return defaultdict(lambda: defaultdict(dict))

    # Find row for each registry dir by matching Model column
    acc_by_dir = {}
    with open(overall) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mdl = row.get("Model", "")
            acc_str = row.get("Overall Acc", "").rstrip("%")
            try:
                acc = float(acc_str) / 100.0
            except ValueError:
                continue
            # Heuristic mapping: find which registry key this row belongs to
            for reg_key, (_canon, _fmt, _scope) in BFCL_MODEL_MAP.items():
                if reg_key.lower() in mdl.lower().replace("-", "").replace(" ", "") or \
                   _row_matches_registry(mdl, reg_key):
                    acc_by_dir.setdefault(reg_key, acc)

    # Fallback: direct textual matching
    acc_by_dir = {}
    with open(overall) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        mdl = row.get("Model", "")
        try:
            acc = float(row.get("Overall Acc", "").rstrip("%")) / 100.0
        except ValueError:
            continue
        reg_key = _map_row_to_registry(mdl)
        if reg_key:
            acc_by_dir[reg_key] = acc

    # Sum tokens per registry dir
    by_model_fmt = defaultdict(lambda: defaultdict(dict))
    # Use "input" scope for BFCL: compressed schemas, native tool-call output.
    # Rationale: "full" scope collapses to ~5% on all formats (parser can't decode
    # TOON/TRON output reliably under AST eval), so compression effects are masked.
    for reg_key, (canon, fmt, scope) in BFCL_MODEL_MAP.items():
        if scope != "input":
            continue
        if reg_key not in acc_by_dir:
            continue
        tokens = _sum_bfcl_tokens(result_dir / reg_key)
        by_model_fmt[canon][fmt]["__agg__"] = {
            "num_tasks": 1,
            "pass_at_1": acc_by_dir[reg_key],
            "tokens": tokens,
            "scope": scope,
        }
    return by_model_fmt


def _map_row_to_registry(model_text):
    t = model_text.lower()
    if "qwen3-32b" in t and "openrouter" in t:
        if "json full" in t: return "qwen3-32b-openrouter-json-full"
        if "toon full" in t: return "qwen3-32b-openrouter-toon-full"
        if "tron full" in t: return "qwen3-32b-openrouter-tron-full"
        if "toon" in t: return "qwen3-32b-openrouter-toon"
        if "tron" in t: return "qwen3-32b-openrouter-tron"
        return "qwen3-32b-openrouter"
    if "qwen3-30b" in t and "local" in t:
        if "json full" in t: return "qwen3-30b-local-json-full"
        if "toon full" in t: return "qwen3-30b-local-toon-full"
        if "tron full" in t: return "qwen3-30b-local-tron-full"
        if "toon" in t: return "qwen3-30b-local-toon"
        if "tron" in t: return "qwen3-30b-local-tron"
        return "qwen3-30b-local"
    return None


def _row_matches_registry(_model_text, _reg_key):
    return False  # unused, kept for structural clarity


def _sum_bfcl_tokens(model_result_dir):
    total = 0
    if not model_result_dir.exists():
        return 0
    for f in model_result_dir.rglob("*.json"):
        try:
            with open(f) as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    total += (d.get("input_token_count") or 0)
                    total += (d.get("output_token_count") or 0)
        except Exception:
            continue
    return total


# ─── StableToolBench ───────────────────────────────────────────────────────
def collect_stb():
    """Pass rate from eval_results; tokens from results/{tag}/G*/*.json rollouts."""
    eval_dir = ROOT / "StableToolBench" / "eval_results"
    results_dir = ROOT / "StableToolBench" / "results"

    # Find all (tag, fmt, tc) directories that have been evaluated
    by_model_fmt = defaultdict(lambda: defaultdict(dict))
    if not eval_dir.exists():
        return by_model_fmt

    pattern = re.compile(r"^(?P<tag>.+)_(?P<fmt>json|toon|tron)_tc(?P<tc>json|toon|tron)$")
    for eval_subdir in eval_dir.iterdir():
        if not eval_subdir.is_dir():
            continue
        m = pattern.match(eval_subdir.name)
        if not m:
            continue
        tag = m.group("tag")
        fmt = m.group("fmt")
        tc = m.group("tc")
        if fmt != tc:
            continue  # only "full" scope

        # Compute pass rate across all groups
        total_solved = 0
        total_trials = 0
        for group_file in eval_subdir.glob("G*_instruction_results.json"):
            try:
                with open(group_file) as fh:
                    data = json.load(fh)
            except Exception:
                continue
            for task_id, info in data.items():
                is_solved = info.get("is_solved", {})
                for trial_id, status in is_solved.items():
                    total_trials += 1
                    if "Solved" in str(status) and "Unsolved" not in str(status):
                        total_solved += 1
        if total_trials == 0:
            continue
        pass_rate = total_solved / total_trials

        # Sum tokens from rollouts
        tokens = _sum_stb_tokens(results_dir / eval_subdir.name)

        canon = canon_model(tag)
        by_model_fmt[canon][fmt]["__agg__"] = {
            "num_tasks": total_trials,
            "pass_rate": pass_rate,
            "tokens": tokens,
        }
    return by_model_fmt


def _sum_stb_tokens(results_subdir):
    total = 0
    if not results_subdir.exists():
        return 0
    for group_dir in results_subdir.glob("G*_instruction"):
        for f in group_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    d = json.load(fh)
            except Exception:
                continue
            ag = d.get("answer_generation")
            if isinstance(ag, dict):
                total += ag.get("total_tokens") or 0
    return total


# ─── Aggregation ───────────────────────────────────────────────────────────
def aggregate_benchmark(raw, metric_key):
    """Collapse per-category data into per-(model, format) totals.

    Returns {model: {fmt: {"tokens": sum, "acc": weighted_avg}}}.
    """
    out = defaultdict(dict)
    for model, fmts in raw.items():
        for fmt, cats in fmts.items():
            total_tok = sum(c.get("tokens", 0) for c in cats.values())
            total_tasks = sum(c.get("num_tasks", 0) for c in cats.values()) or 1
            if metric_key == "pass_at_1":
                w_acc = sum(c.get("pass_at_1", 0) * c.get("num_tasks", 0) for c in cats.values()) / total_tasks
            elif metric_key == "pass_rate":
                w_acc = sum(c.get("pass_rate", 0) * c.get("num_tasks", 0) for c in cats.values()) / total_tasks
            else:
                w_acc = 0
            out[model][fmt] = {"tokens": total_tok, "acc": w_acc}
    return out


def build_final(benchpp, universe, bfcl, stb):
    """For each benchmark, compute (tok_pct, acc_pct) vs JSON same-model, then merge."""
    result = defaultdict(dict)  # model -> fmt -> bench -> dict

    def _deltas(agg, bench_name, metric_key=None):
        for model, fmts in agg.items():
            json_entry = fmts.get("json")
            if not json_entry:
                continue
            json_tok = json_entry["tokens"]
            json_acc = json_entry["acc"]
            for fmt in ("toon", "tron"):
                if fmt not in fmts:
                    continue
                e = fmts[fmt]
                if json_tok > 0:
                    tok_pct = (e["tokens"] - json_tok) / json_tok * 100
                else:
                    tok_pct = None
                if json_acc > 0:
                    acc_pct = (e["acc"] - json_acc) / json_acc * 100
                else:
                    acc_pct = None
                if tok_pct is None or acc_pct is None:
                    continue
                result[model].setdefault(fmt.upper(), {})[bench_name] = {
                    "tok_pct": round(tok_pct, 2),
                    "acc_pct": round(acc_pct, 2),
                    "acc_abs_json": round(json_acc, 4),
                    "acc_abs": round(e["acc"], 4),
                    "tokens_json": json_tok,
                    "tokens": e["tokens"],
                }

    agg_bp = aggregate_benchmark(benchpp, "pass_at_1")
    agg_uni = aggregate_benchmark(universe, "pass_rate")
    agg_bfcl = aggregate_benchmark(bfcl, "pass_at_1")
    agg_stb = aggregate_benchmark(stb, "pass_rate")

    _deltas(agg_bp, "BPP")
    _deltas(agg_uni, "UNI")
    _deltas(agg_bfcl, "BFCL")
    _deltas(agg_stb, "STB")

    return result


def main():
    print("Collecting MCPToolBenchPP...")
    bp = collect_benchpp()
    print(f"  models: {sorted(bp.keys())}")

    print("Collecting MCP-Universe...")
    uni = collect_universe()
    print(f"  models: {sorted(uni.keys())}")

    print("Collecting BFCL...")
    bfcl = collect_bfcl()
    print(f"  models: {sorted(bfcl.keys())}")

    print("Collecting StableToolBench...")
    stb = collect_stb()
    print(f"  models: {sorted(stb.keys())}")

    final = build_final(bp, uni, bfcl, stb)
    with open(OUT_PATH, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved {OUT_PATH}")

    # Summary
    print("\n=== Summary ===")
    for model in sorted(final.keys()):
        fmts = final[model]
        benches = set()
        for fmt, bd in fmts.items():
            benches.update(bd.keys())
        print(f"  {model}: formats={sorted(fmts.keys())} benchmarks={sorted(benches)}")


if __name__ == "__main__":
    main()
