#!/usr/bin/env python3
"""
Result Analyzer for cross-benchmark MCP format compression experiments.

Collects results from all three benchmarks and produces:
  - Per-benchmark accuracy tables (format × model)
  - Token usage comparison tables
  - Cost analysis
  - LaTeX-ready tables for the paper

Usage:
  python analyze_results.py                       # scan default log dirs
  python analyze_results.py --benchpp-logs DIR     # custom MCPToolBenchPP log dir
"""

import argparse
import json
import glob
import csv
import os
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "experiment_logs" / "analysis"


# ── MCPToolBenchPP result parsing ───────────────────────────────────────────

def collect_benchpp_results(log_dir=None):
    """Collect results from MCPToolBenchPP log JSON files."""
    if log_dir is None:
        log_dir = ROOT / "MCPToolBenchPP" / "logs"

    results = []
    for log_file in sorted(Path(log_dir).rglob("*.json")):
        try:
            with open(log_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        run_info = data.get("run_info", {})
        metrics = data.get("metrics", [])
        token_usage = run_info.get("token_usage", {})

        for m in metrics:
            results.append({
                "benchmark": "MCPToolBenchPP",
                "category": run_info.get("category", "?"),
                "model": run_info.get("model", "?"),
                "format": run_info.get("tool_format", "json"),
                "tool_mode": run_info.get("tool_mode", "?"),
                "num_tasks": m.get("num_tasks", 0),
                "pass_at_1": m.get("pass@1", 0),
                "tool_pass_at_1": m.get("tool_pass@1", 0),
                "param_pass_at_1": m.get("parameter_pass@1", 0),
                "total_api_tokens": token_usage.get("total_api_total_tokens", 0),
                "total_model_cost": token_usage.get("total_model_cost_usd", 0),
                "total_schema_tokens": token_usage.get("total_local_mcp_schema_tokens", 0),
                "total_result_tokens": token_usage.get("total_local_tool_result_tokens", 0),
                "total_tc_output_tokens": token_usage.get("total_local_tool_call_output_tokens", 0),
                "source_file": str(log_file),
            })

    return results


# ── MCP-Universe result parsing ─────────────────────────────────────────────

def collect_universe_results(report_dir=None):
    """Collect results from MCP-Universe benchmark reports."""
    if report_dir is None:
        report_dir = ROOT / "MCP-Universe" / "test_full_report"

    results = []
    for report_file in sorted(Path(report_dir).rglob("*.json")):
        try:
            with open(report_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        # Extract format from filename or config
        fname = report_file.stem
        fmt = "json"
        if "_toon" in fname:
            fmt = "toon"
        elif "_tron" in fname:
            fmt = "tron"

        # Extract category from filename
        category = fname.replace("_toon", "").replace("_tron", "")

        # Parse evaluation results
        eval_data = data.get("evaluation", {})
        tasks = data.get("tasks", {})
        n_tasks = len(tasks) if isinstance(tasks, (list, dict)) else 0

        passed = 0
        total = 0
        for task_name, task_data in (tasks.items() if isinstance(tasks, dict) else []):
            evals = task_data.get("evaluation_results", [])
            total += 1
            if all(e.get("passed", False) for e in evals):
                passed += 1

        # Token counts from traces
        token_data = data.get("token_counts", {})

        results.append({
            "benchmark": "MCP-Universe",
            "category": category,
            "model": data.get("model", "?"),
            "format": fmt,
            "num_tasks": total,
            "pass_rate": passed / total if total > 0 else 0,
            "passed": passed,
            "total_schema_tokens": token_data.get("mcp_schema_tokens", 0),
            "total_result_tokens": token_data.get("tool_result_tokens", 0),
            "total_tc_output_tokens": token_data.get("tool_call_output_tokens", 0),
            "source_file": str(report_file),
        })

    return results


# ── mcp-bench result parsing ────────────────────────────────────────────────

def collect_mcpbench_results(results_dir=None):
    """Collect results from mcp-bench output JSON files."""
    if results_dir is None:
        results_dir = ROOT / "mcp-bench" / "results" / "gaia"

    results = []
    for result_file in sorted(Path(results_dir).rglob("*.json")):
        try:
            with open(result_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        # mcp-bench result structure varies; try common patterns
        summary = data.get("summary", data.get("results", {}))
        if isinstance(summary, list):
            for item in summary:
                results.append(_parse_mcpbench_item(item, result_file))
        elif isinstance(summary, dict):
            results.append(_parse_mcpbench_item(summary, result_file))

    return results


def _parse_mcpbench_item(item, source_file):
    """Parse a single mcp-bench result item."""
    return {
        "benchmark": "mcp-bench",
        "model": item.get("model", "?"),
        "format": item.get("compression_format", item.get("format", "?")),
        "tc_format": item.get("tool_call_format", "?"),
        "num_tasks": item.get("total_tasks", 0),
        "success_rate": item.get("success_rate", 0),
        "avg_steps": item.get("avg_steps", 0),
        "total_tokens": item.get("total_tokens", 0),
        "total_cost": item.get("total_cost_usd", 0),
        "source_file": str(source_file),
    }


# ── Table generation ────────────────────────────────────────────────────────

def print_accuracy_table(results, title):
    """Print a format × model accuracy table."""
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")

    # Group by (format, model)
    grouped = defaultdict(list)
    for r in results:
        key = (r.get("format", "?"), r.get("model", "?"))
        grouped[key].append(r)

    formats = sorted(set(r.get("format", "?") for r in results))
    models = sorted(set(r.get("model", "?") for r in results))

    if not formats or not models:
        print("  No data found")
        return

    # Header
    header = f"{'Format':<8}"
    for model in models:
        header += f" {model[:16]:>16}"
    print(header)
    print("-" * len(header))

    for fmt in formats:
        row = f"{fmt:<8}"
        for model in models:
            entries = grouped.get((fmt, model), [])
            if entries:
                # Use pass_at_1 for benchpp, pass_rate for universe, success_rate for mcpbench
                if "pass_at_1" in entries[0]:
                    val = sum(e["pass_at_1"] for e in entries) / len(entries)
                elif "pass_rate" in entries[0]:
                    val = sum(e["pass_rate"] for e in entries) / len(entries)
                elif "success_rate" in entries[0]:
                    val = sum(e["success_rate"] for e in entries) / len(entries)
                else:
                    val = 0
                row += f" {val:>15.3f}"
            else:
                row += f" {'—':>16}"
        print(row)


def print_token_table(results, title):
    """Print token usage comparison table."""
    print(f"\n{'='*70}")
    print(f"{title} — Token Usage")
    print(f"{'='*70}")

    formats = sorted(set(r.get("format", "?") for r in results))

    print(f"{'Format':<8} {'Schema':>10} {'Results':>10} {'TC Output':>10} {'Total API':>12}")
    print(f"{'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

    for fmt in formats:
        entries = [r for r in results if r.get("format") == fmt]
        schema = sum(r.get("total_schema_tokens", 0) for r in entries)
        result_t = sum(r.get("total_result_tokens", 0) for r in entries)
        tc = sum(r.get("total_tc_output_tokens", 0) for r in entries)
        api = sum(r.get("total_api_tokens", 0) for r in entries)
        print(f"{fmt:<8} {schema:>10} {result_t:>10} {tc:>10} {api:>12}")


def generate_latex_table(results, benchmark_name):
    """Generate a LaTeX table for the paper."""
    formats = sorted(set(r.get("format", "?") for r in results))
    models = sorted(set(r.get("model", "?") for r in results))

    lines = []
    lines.append(f"% {benchmark_name} accuracy table")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    cols = "l" + "c" * len(models)
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\toprule")
    header = "Format & " + " & ".join(m[:12] for m in models) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    grouped = defaultdict(list)
    for r in results:
        grouped[(r.get("format"), r.get("model"))].append(r)

    for fmt in formats:
        cells = [fmt.upper()]
        for model in models:
            entries = grouped.get((fmt, model), [])
            if entries:
                if "pass_at_1" in entries[0]:
                    val = sum(e["pass_at_1"] for e in entries) / len(entries)
                elif "pass_rate" in entries[0]:
                    val = sum(e["pass_rate"] for e in entries) / len(entries)
                elif "success_rate" in entries[0]:
                    val = sum(e["success_rate"] for e in entries) / len(entries)
                else:
                    val = 0
                cells.append(f"{val:.3f}")
            else:
                cells.append("—")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{" + benchmark_name + r" accuracy by format and model.}")
    lines.append(r"\label{tab:" + benchmark_name.lower().replace("-", "_").replace(" ", "_") + "}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--benchpp-logs", type=str, default=None)
    parser.add_argument("--universe-reports", type=str, default=None)
    parser.add_argument("--mcpbench-results", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Cross-Benchmark Result Analysis")
    print("=" * 70)

    # Collect results
    benchpp = collect_benchpp_results(args.benchpp_logs)
    universe = collect_universe_results(args.universe_reports)
    mcpbench = collect_mcpbench_results(args.mcpbench_results)

    print(f"\nCollected: {len(benchpp)} MCPToolBenchPP, {len(universe)} MCP-Universe, {len(mcpbench)} mcp-bench")

    # Print accuracy tables
    if benchpp:
        print_accuracy_table(benchpp, "MCPToolBenchPP — Pass@1")
        print_token_table(benchpp, "MCPToolBenchPP")

    if universe:
        print_accuracy_table(universe, "MCP-Universe — Pass Rate")
        print_token_table(universe, "MCP-Universe")

    if mcpbench:
        print_accuracy_table(mcpbench, "mcp-bench — Success Rate")

    # Generate LaTeX tables
    latex_parts = []
    if benchpp:
        latex_parts.append(generate_latex_table(benchpp, "MCPToolBenchPP"))
    if universe:
        latex_parts.append(generate_latex_table(universe, "MCP-Universe"))
    if mcpbench:
        latex_parts.append(generate_latex_table(mcpbench, "mcp-bench"))

    if latex_parts:
        latex_path = OUTPUT_DIR / "paper_tables.tex"
        with open(latex_path, "w") as f:
            f.write("\n\n".join(latex_parts))
        print(f"\nLaTeX tables saved to: {latex_path}")

    # Save combined results as JSON
    combined = {
        "benchpp": benchpp,
        "universe": universe,
        "mcpbench": mcpbench,
    }
    combined_path = OUTPUT_DIR / "combined_results.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Combined results saved to: {combined_path}")


if __name__ == "__main__":
    main()
