#!/usr/bin/env python3
"""
Result Analyzer for cross-benchmark MCP format compression experiments.

Collects results from MCPToolBenchPP and MCP-Universe and produces:
  - Per-benchmark accuracy tables (format × category)
  - Token usage comparison tables
  - Cost analysis
  - LaTeX-ready tables for the paper

Usage:
  python analyze_results.py                        # scan default log dirs
  python analyze_results.py --exp2-only            # only exp2 results (March 2026+)
  python analyze_results.py --benchpp-logs DIR      # custom MCPToolBenchPP log dir
"""

import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "experiment_logs" / "analysis"

# Expected task counts per category for filtering valid runs
BENCHPP_TASK_COUNTS = {
    "finance": 90, "pay": 310, "search": 181,
    "file_system": 241, "map": 500, "browser": 187,
}

# Categories excluded from pass@1 analysis due to external API failures
# (Tavily API quota exhausted during TOON/TRON runs — 842/846 calls returned HTTP 432)
BENCHPP_EXCLUDED_CATEGORIES = {"search", "browser"}
UNIVERSE_CATEGORIES = [
    "financial_analysis", "location_navigation", "browser_automation",
    "repository_management", "3d_design",
]


# ── MCPToolBenchPP result parsing ───────────────────────────────────────────

def collect_benchpp_results(log_dir=None, exp2_only=False, model_filter="qwen3-32b"):
    """Collect results from MCPToolBenchPP log JSON files."""
    if log_dir is None:
        log_dir = ROOT / "MCPToolBenchPP" / "logs"

    results = []
    seen = {}  # (category, format) -> result, keep latest

    for log_file in sorted(Path(log_dir).rglob("*.json")):
        try:
            with open(log_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        run_info = data.get("run_info", {})
        metrics = data.get("metrics", [])
        token_usage = run_info.get("token_usage", {})

        if not metrics:
            continue

        m = metrics[0]
        num_tasks = m.get("num_tasks", 0)
        category = run_info.get("category", "?")
        fmt = run_info.get("tool_format", "json")
        model = run_info.get("model", "?")

        # Filter: skip smoke tests (small task counts)
        if num_tasks < 10:
            continue

        # Filter: only include runs from the specified model
        if model_filter and model != model_filter:
            continue

        # Filter: exp2 only = files from March 2026+ (shared_format integration)
        if exp2_only:
            fname = log_file.name
            date_match = re.search(r'_(\d{8})_', fname)
            if date_match and int(date_match.group(1)) < 20260302:
                continue

        result = {
            "benchmark": "MCPToolBenchPP",
            "category": category,
            "model": model,
            "format": fmt,
            "num_tasks": num_tasks,
            "pass_at_1": m.get("pass@1", 0),
            "tool_pass_at_1": m.get("tool_pass@1", 0),
            "param_pass_at_1": m.get("parameter_pass@1", 0),
            "num_trials": m.get("num_trials_total", 0),
            "num_passed": m.get("num_passed_total", 0),
            "total_api_prompt_tokens": token_usage.get("total_api_prompt_tokens", 0),
            "total_api_completion_tokens": token_usage.get("total_api_completion_tokens", 0),
            "total_api_tokens": token_usage.get("total_api_total_tokens", 0),
            "total_model_cost": token_usage.get("total_model_cost_usd", 0),
            "total_judge_cost": token_usage.get("total_judge_cost_usd", 0),
            "total_schema_tokens": token_usage.get("total_local_mcp_schema_tokens", 0),
            "total_result_tokens": token_usage.get("total_local_tool_result_tokens", 0),
            "total_tc_output_tokens": token_usage.get("total_local_tool_call_output_tokens", 0),
            "source_file": str(log_file),
        }

        # Keep the latest run per (category, format, model)
        key = (category, fmt, model)
        seen[key] = result

    # Exclude categories with known external API failures
    return [r for r in seen.values() if r["category"] not in BENCHPP_EXCLUDED_CATEGORIES]


# ── MCP-Universe result parsing ─────────────────────────────────────────────

def parse_universe_report(report_path):
    """Parse a MCP-Universe markdown report file into structured data."""
    with open(report_path) as f:
        content = f.read()

    # Extract format from filename
    fname = Path(report_path).name
    if "_tron_" in fname:
        fmt = "tron"
    elif "_toon_" in fname:
        fmt = "toon"
    elif "_json_" in fname:
        fmt = "json"
    else:
        fmt = "json"

    # Extract category from filename (everything before format marker)
    category = re.sub(r'_(json|toon|tron)_tc(json|toon|tron)_.*', '', fname)

    # Parse markdown table rows
    tasks = []
    for line in content.split('\n'):
        if not line.startswith('|**'):
            continue
        # Parse: |**name**:| passed | not_passed | score | llm_calls | format | prompt | compl | total | cost | schema | call | result |
        parts = [p.strip() for p in line.split('|')]
        parts = [p for p in parts if p]  # remove empty strings
        if len(parts) < 13:
            continue

        try:
            task_name = re.sub(r'\*\*|:', '', parts[0]).strip()
            passed = int(parts[1])
            not_passed = int(parts[2])
            score = float(parts[3])
            llm_calls = int(parts[4])
            prompt_tok = int(parts[6])
            compl_tok = int(parts[7])
            total_tok = int(parts[8])
            cost_str = parts[9].replace('$', '')
            cost = float(cost_str)
            schema_tok = int(parts[10])
            call_tok = int(parts[11])
            result_tok = int(parts[12])

            tasks.append({
                "name": task_name,
                "passed": passed,
                "not_passed": not_passed,
                "score": score,
                "llm_calls": llm_calls,
                "prompt_tokens": prompt_tok,
                "completion_tokens": compl_tok,
                "total_tokens": total_tok,
                "cost_usd": cost,
                "schema_tokens": schema_tok,
                "call_tokens": call_tok,
                "result_tokens": result_tok,
            })
        except (ValueError, IndexError):
            continue

    if not tasks:
        return None

    # Aggregate
    total_tasks = len(tasks)
    tasks_passed = sum(1 for t in tasks if t["not_passed"] == 0 and t["passed"] > 0)
    pass_rate = tasks_passed / total_tasks if total_tasks > 0 else 0

    return {
        "benchmark": "MCP-Universe",
        "category": category,
        "format": fmt,
        "model": "qwen3-32b",
        "num_tasks": total_tasks,
        "passed": tasks_passed,
        "pass_rate": pass_rate,
        "total_llm_calls": sum(t["llm_calls"] for t in tasks),
        "avg_llm_calls": sum(t["llm_calls"] for t in tasks) / total_tasks,
        "total_prompt_tokens": sum(t["prompt_tokens"] for t in tasks),
        "total_completion_tokens": sum(t["completion_tokens"] for t in tasks),
        "total_api_tokens": sum(t["total_tokens"] for t in tasks),
        "total_cost": sum(t["cost_usd"] for t in tasks),
        "total_schema_tokens": sum(t["schema_tokens"] for t in tasks),
        "total_tc_output_tokens": sum(t["call_tokens"] for t in tasks),
        "total_result_tokens": sum(t["result_tokens"] for t in tasks),
        "tasks": tasks,
        "source_file": str(report_path),
    }


def collect_universe_results(report_dir=None, exp2_only=False):
    """Collect results from MCP-Universe benchmark reports."""
    if report_dir is None:
        report_dir = ROOT / "MCP-Universe" / "log" / "test_full_report"

    results = []
    report_path = Path(report_dir)
    if not report_path.exists():
        return results

    for entry in sorted(report_path.iterdir()):
        # Only process exp2 reports (with qwen3-32b suffix)
        if exp2_only and "qwen3-32b" not in entry.name:
            continue
        # Skip web_search
        if "web_search" in entry.name:
            continue
        if not entry.is_file():
            continue

        result = parse_universe_report(entry)
        if result:
            results.append(result)

    return results


# ── Table printing ──────────────────────────────────────────────────────────

def pct_change(val, baseline):
    """Format percentage change relative to baseline."""
    if baseline == 0:
        return ""
    pct = (val - baseline) / baseline * 100
    if pct > 0:
        return f" (+{pct:.1f}%)"
    else:
        return f" ({pct:.1f}%)"


def print_benchpp_accuracy(results):
    """Print MCPToolBenchPP accuracy table by format × category."""
    print(f"\n{'='*90}")
    print("MCPToolBenchPP — Accuracy (pass@1 / tool_pass@1 / param_pass@1)")
    print(f"{'='*90}")

    formats = ["json", "toon", "tron"]
    categories = sorted(set(r["category"] for r in results))

    # Per-category breakdown
    print(f"\n{'Category':<14}", end="")
    for fmt in formats:
        print(f" {'pass@1':>8} {'tool':>6} {'param':>6}", end="")
    print()
    print("-" * 90)

    for cat in categories:
        print(f"{cat:<14}", end="")
        for fmt in formats:
            entries = [r for r in results if r["category"] == cat and r["format"] == fmt]
            if entries:
                r = entries[0]
                print(f" {r['pass_at_1']:>8.3f} {r['tool_pass_at_1']:>6.3f} {r['param_pass_at_1']:>6.3f}", end="")
            else:
                print(f" {'—':>8} {'—':>6} {'—':>6}", end="")
        print()

    # Weighted average
    print("-" * 90)
    print(f"{'Weighted Avg':<14}", end="")
    for fmt in formats:
        entries = [r for r in results if r["format"] == fmt]
        if entries:
            total_tasks = sum(r["num_tasks"] for r in entries)
            w_pass = sum(r["pass_at_1"] * r["num_tasks"] for r in entries) / total_tasks
            w_tool = sum(r["tool_pass_at_1"] * r["num_tasks"] for r in entries) / total_tasks
            w_param = sum(r["param_pass_at_1"] * r["num_tasks"] for r in entries) / total_tasks
            print(f" {w_pass:>8.3f} {w_tool:>6.3f} {w_param:>6.3f}", end="")
        else:
            print(f" {'—':>8} {'—':>6} {'—':>6}", end="")
    print()


def print_universe_accuracy(results):
    """Print MCP-Universe accuracy table by format × category."""
    print(f"\n{'='*80}")
    print("MCP-Universe — Pass Rate")
    print(f"{'='*80}")

    formats = ["json", "toon", "tron"]
    categories = sorted(set(r["category"] for r in results))

    print(f"\n{'Category':<25}", end="")
    for fmt in formats:
        print(f" {fmt:>12}", end="")
    print()
    print("-" * 65)

    for cat in categories:
        print(f"{cat:<25}", end="")
        for fmt in formats:
            entries = [r for r in results if r["category"] == cat and r["format"] == fmt]
            if entries:
                r = entries[0]
                print(f" {r['passed']:>3}/{r['num_tasks']:<3} ({r['pass_rate']:.1%})", end="")
            else:
                print(f" {'—':>12}", end="")
        print()

    # Overall
    print("-" * 65)
    print(f"{'Overall':<25}", end="")
    for fmt in formats:
        entries = [r for r in results if r["format"] == fmt]
        if entries:
            total_passed = sum(r["passed"] for r in entries)
            total_tasks = sum(r["num_tasks"] for r in entries)
            rate = total_passed / total_tasks if total_tasks > 0 else 0
            print(f" {total_passed:>3}/{total_tasks:<3} ({rate:.1%})", end="")
        else:
            print(f" {'—':>12}", end="")
    print()


def print_token_table(results, title):
    """Print token usage comparison table across formats."""
    print(f"\n{'='*90}")
    print(f"{title} — Token Usage")
    print(f"{'='*90}")

    formats = ["json", "toon", "tron"]
    json_totals = {}

    print(f"\n{'Component':<18}", end="")
    for fmt in formats:
        print(f" {fmt.upper():>16}", end="")
    print()
    print("-" * 66)

    components = [
        ("Schema Tokens", "total_schema_tokens"),
        ("TC Output Tokens", "total_tc_output_tokens"),
        ("Result Tokens", "total_result_tokens"),
        ("API Prompt Tok", "total_api_prompt_tokens"),
        ("API Compl Tok", "total_api_completion_tokens"),
        ("API Total Tok", "total_api_tokens"),
    ]

    for label, key in components:
        vals = {}
        for fmt in formats:
            entries = [r for r in results if r.get("format") == fmt]
            vals[fmt] = sum(r.get(key, 0) for r in entries)

        print(f"{label:<18}", end="")
        for fmt in formats:
            v = vals[fmt]
            if v == 0:
                print(f" {'—':>16}", end="")
            elif fmt == "json":
                print(f" {v:>16,}", end="")
            else:
                change = pct_change(v, vals.get("json", 0))
                print(f" {v:>10,}{change:>6}", end="")
        print()

    # Cost
    print("-" * 66)
    for fmt in formats:
        entries = [r for r in results if r.get("format") == fmt]
        cost = sum(r.get("total_model_cost", 0) + r.get("total_cost", 0) for r in entries)
        if fmt == "json":
            print(f"{'Cost (USD)':<18} ${cost:>15.2f}", end="")
        else:
            json_entries = [r for r in results if r.get("format") == "json"]
            json_cost = sum(r.get("total_model_cost", 0) + r.get("total_cost", 0) for r in json_entries)
            change = pct_change(cost, json_cost)
            print(f" ${cost:>9.2f}{change:>6}", end="")
    print()


def print_universe_execution_stats(results):
    """Print MCP-Universe execution statistics."""
    print(f"\n{'='*80}")
    print("MCP-Universe — Execution Statistics")
    print(f"{'='*80}")

    formats = ["json", "toon", "tron"]

    print(f"\n{'Metric':<22}", end="")
    for fmt in formats:
        print(f" {fmt.upper():>16}", end="")
    print()
    print("-" * 70)

    metrics = [
        ("Total LLM Calls", lambda rs: sum(r.get("total_llm_calls", 0) for r in rs)),
        ("Avg Calls/Task", lambda rs: sum(r.get("avg_llm_calls", 0) * r.get("num_tasks", 0) for r in rs) / max(sum(r.get("num_tasks", 0) for r in rs), 1)),
        ("Total Cost (USD)", lambda rs: sum(r.get("total_cost", 0) for r in rs)),
    ]

    for label, fn in metrics:
        json_val = fn([r for r in results if r["format"] == "json"])
        for fmt in formats:
            entries = [r for r in results if r["format"] == fmt]
            val = fn(entries)
            if fmt == "json":
                if "Cost" in label:
                    print(f"{label:<22} ${val:>15.2f}", end="")
                elif "Avg" in label:
                    print(f"{label:<22} {val:>16.1f}", end="")
                else:
                    print(f"{label:<22} {val:>16,}", end="")
            else:
                change = pct_change(val, json_val)
                if "Cost" in label:
                    print(f" ${val:>9.2f}{change:>6}", end="")
                elif "Avg" in label:
                    print(f" {val:>10.1f}{change:>6}", end="")
                else:
                    print(f" {val:>10,}{change:>6}", end="")
        print()


# ── LaTeX table generation ──────────────────────────────────────────────────

def generate_benchpp_latex(results):
    """Generate LaTeX accuracy table for MCPToolBenchPP."""
    formats = ["json", "toon", "tron"]
    categories = sorted(set(r["category"] for r in results))

    lines = [
        r"% MCPToolBenchPP accuracy table (exp2: full compression)",
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \footnotesize",
        r"    \caption{MCPToolBenchPP accuracy by format and category (Qwen3-32B, full compression). Values are pass@1 estimated over 5 trials.}",
        r"    \label{tab:benchpp_accuracy}",
        r"    \begin{tabular}{@{}l*{3}{rrr}@{}}",
        r"        \toprule",
        r"        & \multicolumn{3}{c}{\textbf{JSON}} & \multicolumn{3}{c}{\textbf{TOON}} & \multicolumn{3}{c}{\textbf{TRON}} \\",
        r"        \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}",
        r"        \textbf{Category} & pass@1 & tool & param & pass@1 & tool & param & pass@1 & tool & param \\",
        r"        \midrule",
    ]

    for cat in categories:
        row = f"        {cat.replace('_', ' ').title()}"
        for fmt in formats:
            entries = [r for r in results if r["category"] == cat and r["format"] == fmt]
            if entries:
                r = entries[0]
                row += f" & {r['pass_at_1']:.3f} & {r['tool_pass_at_1']:.3f} & {r['param_pass_at_1']:.3f}"
            else:
                row += r" & --- & --- & ---"
        row += r" \\"
        lines.append(row)

    # Weighted average
    lines.append(r"        \midrule")
    row = r"        \textbf{Weighted Avg}"
    for fmt in formats:
        entries = [r for r in results if r["format"] == fmt]
        if entries:
            total_tasks = sum(r["num_tasks"] for r in entries)
            w_pass = sum(r["pass_at_1"] * r["num_tasks"] for r in entries) / total_tasks
            w_tool = sum(r["tool_pass_at_1"] * r["num_tasks"] for r in entries) / total_tasks
            w_param = sum(r["param_pass_at_1"] * r["num_tasks"] for r in entries) / total_tasks
            row += f" & \\textbf{{{w_pass:.3f}}} & \\textbf{{{w_tool:.3f}}} & \\textbf{{{w_param:.3f}}}"
    row += r" \\"
    lines.append(row)

    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table*}",
    ])

    return "\n".join(lines)


def generate_universe_latex(results):
    """Generate LaTeX accuracy table for MCP-Universe."""
    formats = ["json", "toon", "tron"]
    categories = sorted(set(r["category"] for r in results))

    lines = [
        r"% MCP-Universe accuracy table (exp2: full compression)",
        r"\begin{table}[t]",
        r"    \centering",
        r"    \footnotesize",
        r"    \caption{MCP-Universe pass rate by format and category (Qwen3-32B, full compression).}",
        r"    \label{tab:universe_accuracy}",
        r"    \begin{tabular}{@{}lrrr@{}}",
        r"        \toprule",
        r"        \textbf{Category} & \textbf{JSON} & \textbf{TOON} & \textbf{TRON} \\",
        r"        \midrule",
    ]

    for cat in categories:
        cat_label = cat.replace("_", " ").title()
        row = f"        {cat_label}"
        for fmt in formats:
            entries = [r for r in results if r["category"] == cat and r["format"] == fmt]
            if entries:
                r = entries[0]
                row += f" & {r['pass_rate']:.1%}"
            else:
                row += r" & ---"
        row += r" \\"
        lines.append(row)

    # Overall
    lines.append(r"        \midrule")
    row = r"        \textbf{Overall}"
    for fmt in formats:
        entries = [r for r in results if r["format"] == fmt]
        if entries:
            total_passed = sum(r["passed"] for r in entries)
            total_tasks = sum(r["num_tasks"] for r in entries)
            rate = total_passed / total_tasks if total_tasks > 0 else 0
            row += f" & \\textbf{{{rate:.1%}}}"
    row += r" \\"
    lines.append(row)

    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_token_latex(benchpp, universe):
    """Generate combined token usage LaTeX table."""
    formats = ["json", "toon", "tron"]
    all_results = benchpp + universe

    lines = [
        r"% Token usage table (combined benchmarks)",
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \footnotesize",
        r"    \caption{Token consumption across formats (combined MCPToolBenchPP + MCP-Universe). Percentages relative to JSON baseline.}",
        r"    \label{tab:token_usage}",
        r"    \begin{tabular}{@{}lrrr@{}}",
        r"        \toprule",
        r"        \textbf{Component} & \textbf{JSON} & \textbf{TOON} & \textbf{TRON} \\",
        r"        \midrule",
    ]

    components = [
        ("Schema Tokens", "total_schema_tokens"),
        ("Tool Call Tokens", "total_tc_output_tokens"),
        ("Result Tokens", "total_result_tokens"),
    ]

    for label, key in components:
        json_val = sum(r.get(key, 0) for r in all_results if r["format"] == "json")
        row = f"        {label} & {json_val:,}"
        for fmt in ["toon", "tron"]:
            val = sum(r.get(key, 0) for r in all_results if r["format"] == fmt)
            pct = (val - json_val) / json_val * 100 if json_val > 0 else 0
            color = "red" if pct > 0 else "blue"
            sign = "+" if pct > 0 else ""
            row += f" & {val:,} {{\\scriptsize\\textcolor{{{color}}}{{{sign}{pct:.1f}\\%}}}}"
        row += r" \\"
        lines.append(row)

    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table*}",
    ])

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--benchpp-logs", type=str, default=None)
    parser.add_argument("--universe-reports", type=str, default=None)
    parser.add_argument("--exp2-only", action="store_true", default=True,
                        help="Only include exp2 results (March 2026+)")
    parser.add_argument("--all-runs", action="store_true",
                        help="Include all runs, not just exp2")
    args = parser.parse_args()

    if args.all_runs:
        args.exp2_only = False

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 90)
    print("Cross-Benchmark Result Analysis")
    print(f"Filter: {'exp2 only (March 2026+)' if args.exp2_only else 'all runs'}")
    print("=" * 90)

    # Collect results
    benchpp = collect_benchpp_results(args.benchpp_logs, exp2_only=args.exp2_only)
    universe = collect_universe_results(args.universe_reports, exp2_only=args.exp2_only)

    print(f"\nCollected: {len(benchpp)} MCPToolBenchPP runs, {len(universe)} MCP-Universe runs")

    if benchpp:
        bp_cats = set(r["category"] for r in benchpp)
        bp_fmts = set(r["format"] for r in benchpp)
        print(f"  BenchPP: {len(bp_cats)} categories × {len(bp_fmts)} formats = {len(benchpp)} runs")

    if universe:
        u_cats = set(r["category"] for r in universe)
        u_fmts = set(r["format"] for r in universe)
        print(f"  Universe: {len(u_cats)} categories × {len(u_fmts)} formats = {len(universe)} runs")

    # Print tables
    if benchpp:
        print_benchpp_accuracy(benchpp)
        print_token_table(benchpp, "MCPToolBenchPP")

    if universe:
        print_universe_accuracy(universe)
        print_token_table(universe, "MCP-Universe")
        print_universe_execution_stats(universe)

    # Generate LaTeX tables
    latex_parts = []
    if benchpp:
        latex_parts.append(generate_benchpp_latex(benchpp))
    if universe:
        latex_parts.append(generate_universe_latex(universe))
    if benchpp and universe:
        latex_parts.append(generate_token_latex(benchpp, universe))

    if latex_parts:
        latex_path = OUTPUT_DIR / "paper_tables.tex"
        with open(latex_path, "w") as f:
            f.write("\n\n".join(latex_parts))
        print(f"\nLaTeX tables saved to: {latex_path}")

    # Save combined results as JSON (strip per-task details for universe)
    combined = {
        "benchpp": benchpp,
        "universe": [{k: v for k, v in r.items() if k != "tasks"} for r in universe],
    }
    combined_path = OUTPUT_DIR / "combined_results.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Combined results saved to: {combined_path}")


if __name__ == "__main__":
    main()
