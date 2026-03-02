#!/usr/bin/env python3
"""
Exp 0: Offline Token Analysis

Counts tokens for tool schemas, example tool calls, and prompt components
across JSON, TOON, and TRON formats. Produces tables for the paper without
making any API calls.

Uses data from:
  - MCPToolBenchPP task files (inline tool schemas)
  - shared_format prompt snippets (format explanations, intros, reminders)

Output: CSV + console tables with token counts and compression ratios.
"""

import json
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

# Add shared_format to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "shared_format" / "src"))
import shared_format
from shared_format.converter import ToolFormat
from shared_format.prompt_snippets import (
    get_format_name, get_format_explanation, get_format_intro,
    get_format_reminder, serialize_tools
)
from shared_format.token_counter import count_tokens

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "experiment_logs" / "token_analysis"

FORMATS = [ToolFormat.JSON, ToolFormat.TOON, ToolFormat.TRON]

# MCPToolBenchPP data files
BENCHPP_DATA = {
    "finance": ROOT / "MCPToolBenchPP" / "data" / "finance" / "finance_0724_single_v3.json",
    "pay": ROOT / "MCPToolBenchPP" / "data" / "pay" / "pay_0723_single.json",
    "search": ROOT / "MCPToolBenchPP" / "data" / "search" / "search_0725_single_v2.json",
    "file_system": ROOT / "MCPToolBenchPP" / "data" / "file_system" / "filesystem_0723_single.json",
    "map": ROOT / "MCPToolBenchPP" / "data" / "map" / "map_0717_single_multi_lang_500.json",
    "browser": ROOT / "MCPToolBenchPP" / "data" / "browser" / "browser_0724_single_v3.json",
}


def load_benchpp_tools(data_path):
    """Load tool schemas from a MCPToolBenchPP data file."""
    if not data_path.exists():
        return []
    with open(data_path) as f:
        data = json.load(f)
    all_tools = []
    for item in data:
        tools = json.loads(item["tools"]) if isinstance(item["tools"], str) else item["tools"]
        all_tools.append(tools)
    return all_tools


def analyze_prompt_components():
    """Analyze token counts for prompt building blocks."""
    print("\n" + "=" * 70)
    print("1. Prompt Component Token Analysis")
    print("=" * 70)

    rows = []
    for fmt in FORMATS:
        name = get_format_name(fmt)
        explanation = get_format_explanation(fmt)
        intro = get_format_intro(fmt)
        reminder = get_format_reminder(fmt)

        rows.append({
            "format": name,
            "explanation_tokens": count_tokens(explanation),
            "intro_tokens": count_tokens(intro),
            "reminder_tokens": count_tokens(reminder),
        })

    # Print table
    print(f"\n{'Format':<8} {'Explanation':>13} {'Intro':>8} {'Reminder':>10}")
    print(f"{'-'*8} {'-'*13} {'-'*8} {'-'*10}")
    for r in rows:
        print(f"{r['format']:<8} {r['explanation_tokens']:>13} {r['intro_tokens']:>8} {r['reminder_tokens']:>10}")

    return rows


def analyze_tool_schemas():
    """Analyze token counts for tool schema serialization across formats."""
    print("\n" + "=" * 70)
    print("2. Tool Schema Token Analysis (MCPToolBenchPP)")
    print("=" * 70)

    all_rows = []
    summary = defaultdict(lambda: {"total_json": 0, "total_toon": 0, "total_tron": 0, "count": 0})

    for category, data_path in BENCHPP_DATA.items():
        tool_sets = load_benchpp_tools(data_path)
        if not tool_sets:
            print(f"  {category}: no data found")
            continue

        for i, tools in enumerate(tool_sets):
            row = {"category": category, "task_idx": i, "num_tools": len(tools)}

            for fmt in FORMATS:
                serialized = serialize_tools(tools, fmt)
                tokens = count_tokens(serialized)
                row[f"{fmt.value}_tokens"] = tokens

            # Compression ratios
            json_t = row["json_tokens"]
            if json_t > 0:
                row["toon_ratio"] = row["toon_tokens"] / json_t
                row["tron_ratio"] = row["tron_tokens"] / json_t
            else:
                row["toon_ratio"] = 1.0
                row["tron_ratio"] = 1.0

            all_rows.append(row)
            summary[category]["total_json"] += row["json_tokens"]
            summary[category]["total_toon"] += row["toon_tokens"]
            summary[category]["total_tron"] += row["tron_tokens"]
            summary[category]["count"] += 1

    # Print summary per category
    print(f"\n{'Category':<14} {'Tasks':>5} {'JSON avg':>10} {'TOON avg':>10} {'TRON avg':>10} {'TOON %':>8} {'TRON %':>8}")
    print(f"{'-'*14} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    grand = {"json": 0, "toon": 0, "tron": 0, "count": 0}
    for cat in BENCHPP_DATA:
        s = summary[cat]
        if s["count"] == 0:
            continue
        n = s["count"]
        j_avg = s["total_json"] / n
        o_avg = s["total_toon"] / n
        r_avg = s["total_tron"] / n
        o_pct = (o_avg / j_avg * 100) if j_avg > 0 else 100
        r_pct = (r_avg / j_avg * 100) if j_avg > 0 else 100
        print(f"{cat:<14} {n:>5} {j_avg:>10.1f} {o_avg:>10.1f} {r_avg:>10.1f} {o_pct:>7.1f}% {r_pct:>7.1f}%")
        grand["json"] += s["total_json"]
        grand["toon"] += s["total_toon"]
        grand["tron"] += s["total_tron"]
        grand["count"] += n

    if grand["count"] > 0:
        n = grand["count"]
        j_avg = grand["json"] / n
        o_avg = grand["toon"] / n
        r_avg = grand["tron"] / n
        o_pct = (o_avg / j_avg * 100) if j_avg > 0 else 100
        r_pct = (r_avg / j_avg * 100) if j_avg > 0 else 100
        print(f"{'-'*14} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
        print(f"{'OVERALL':<14} {n:>5} {j_avg:>10.1f} {o_avg:>10.1f} {r_avg:>10.1f} {o_pct:>7.1f}% {r_pct:>7.1f}%")

    return all_rows


def analyze_example_tool_calls():
    """Analyze token counts for example tool call structures."""
    print("\n" + "=" * 70)
    print("3. Example Tool Call Token Analysis")
    print("=" * 70)

    # Representative tool call examples
    examples = {
        "simple": {
            "name": "get_weather",
            "arguments": {"city": "Vienna", "units": "metric"}
        },
        "medium": {
            "name": "search_documents",
            "arguments": {
                "query": "machine learning benchmarks",
                "filters": {"type": "paper", "year_min": 2023},
                "max_results": 10,
                "include_abstracts": True
            }
        },
        "complex": {
            "name": "create_chart",
            "arguments": {
                "title": "Token Compression Ratio by Format",
                "type": "bar",
                "data": {
                    "labels": ["JSON", "TOON", "TRON"],
                    "datasets": [
                        {"label": "Schema", "values": [100, 82, 61]},
                        {"label": "Results", "values": [100, 78, 65]}
                    ]
                },
                "options": {"legend": True, "grid": True, "width": 800, "height": 600}
            }
        }
    }

    rows = []
    print(f"\n{'Example':<12} {'JSON':>8} {'TOON':>8} {'TRON':>8} {'TOON %':>8} {'TRON %':>8}")
    print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, example in examples.items():
        row = {"example": name}
        for fmt in FORMATS:
            serialized = shared_format.serialize(example, fmt)
            tokens = count_tokens(serialized)
            row[f"{fmt.value}_tokens"] = tokens

        j = row["json_tokens"]
        o_pct = (row["toon_tokens"] / j * 100) if j > 0 else 100
        r_pct = (row["tron_tokens"] / j * 100) if j > 0 else 100
        row["toon_pct"] = o_pct
        row["tron_pct"] = r_pct
        print(f"{name:<12} {row['json_tokens']:>8} {row['toon_tokens']:>8} {row['tron_tokens']:>8} {o_pct:>7.1f}% {r_pct:>7.1f}%")
        rows.append(row)

    return rows


def analyze_tool_results():
    """Analyze token counts for sample tool result serialization."""
    print("\n" + "=" * 70)
    print("4. Tool Result Token Analysis")
    print("=" * 70)

    # Representative tool results
    results = {
        "simple_text": {"status": "success", "result": "The current temperature in Vienna is 15°C with partly cloudy skies."},
        "json_data": {
            "status": "success",
            "result": {
                "symbol": "AAPL",
                "price": 189.45,
                "change": -2.30,
                "change_percent": -1.20,
                "volume": 52341200,
                "market_cap": "2.95T",
                "pe_ratio": 29.8
            }
        },
        "list_data": {
            "status": "success",
            "results": [
                {"title": "Paper A", "authors": ["Smith", "Jones"], "year": 2024, "citations": 45},
                {"title": "Paper B", "authors": ["Chen", "Wang"], "year": 2023, "citations": 120},
                {"title": "Paper C", "authors": ["Mueller"], "year": 2024, "citations": 8},
            ]
        }
    }

    rows = []
    print(f"\n{'Result Type':<14} {'JSON':>8} {'TOON':>8} {'TRON':>8} {'TOON %':>8} {'TRON %':>8}")
    print(f"{'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, result in results.items():
        row = {"result_type": name}
        for fmt in FORMATS:
            serialized = shared_format.serialize(result, fmt)
            tokens = count_tokens(serialized)
            row[f"{fmt.value}_tokens"] = tokens

        j = row["json_tokens"]
        o_pct = (row["toon_tokens"] / j * 100) if j > 0 else 100
        r_pct = (row["tron_tokens"] / j * 100) if j > 0 else 100
        print(f"{name:<14} {row['json_tokens']:>8} {row['toon_tokens']:>8} {row['tron_tokens']:>8} {o_pct:>7.1f}% {r_pct:>7.1f}%")
        rows.append(row)

    return rows


def save_csv(rows, filename, fieldnames):
    """Save rows to CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = OUTPUT_DIR / filename
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


def main():
    print("=" * 70)
    print("Exp 0: Offline Token Analysis")
    print("=" * 70)

    # 1. Prompt components
    prompt_rows = analyze_prompt_components()
    save_csv(prompt_rows, "prompt_components.csv",
             ["format", "explanation_tokens", "intro_tokens", "reminder_tokens"])

    # 2. Tool schemas
    schema_rows = analyze_tool_schemas()
    if schema_rows:
        save_csv(schema_rows, "tool_schemas.csv",
                 ["category", "task_idx", "num_tools", "json_tokens", "toon_tokens",
                  "tron_tokens", "toon_ratio", "tron_ratio"])

    # 3. Tool calls
    call_rows = analyze_example_tool_calls()
    save_csv(call_rows, "tool_calls.csv",
             ["example", "json_tokens", "toon_tokens", "tron_tokens", "toon_pct", "tron_pct"])

    # 4. Tool results
    result_rows = analyze_tool_results()
    save_csv(result_rows, "tool_results.csv",
             ["result_type", "json_tokens", "toon_tokens", "tron_tokens"])

    print(f"\n{'='*70}")
    print(f"Token analysis complete. CSVs in: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
