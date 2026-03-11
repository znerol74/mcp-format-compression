#!/usr/bin/env python3
"""Generate paper figures from experiment results."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "experiment_logs" / "analysis" / "combined_results.json"
OUT_DIR = Path(__file__).resolve().parent

# Colors and markers
COLORS = {"toon": "#E74C3C", "tron": "#2980B9"}
MARKERS = {"MCPToolBenchPP": "o", "MCP-Universe": "^"}
FORMAT_LABELS = {"toon": "TOON", "tron": "TRON"}
BENCH_LABELS = {"MCPToolBenchPP": "BenchPP", "MCP-Universe": "Universe"}


def load_data():
    with open(DATA_PATH) as f:
        raw = json.load(f)

    points = []  # list of dicts with benchmark, category, format, token_pct, acc_pct, acc_abs

    # BenchPP: compute per-category % change vs JSON
    benchpp = raw["benchpp"]
    json_by_cat = {r["category"]: r for r in benchpp if r["format"] == "json"}
    for r in benchpp:
        if r["format"] == "json":
            continue
        cat = r["category"]
        jref = json_by_cat.get(cat)
        if not jref:
            continue
        tok_pct = (r["total_api_tokens"] - jref["total_api_tokens"]) / jref["total_api_tokens"] * 100
        acc_pct = (r["pass_at_1"] - jref["pass_at_1"]) / jref["pass_at_1"] * 100 if jref["pass_at_1"] > 0 else 0
        points.append({
            "benchmark": "MCPToolBenchPP",
            "category": cat,
            "format": r["format"],
            "token_pct": tok_pct,
            "acc_pct": acc_pct,
            "acc_abs": r["pass_at_1"],
            "num_tasks": r["num_tasks"],
        })

    # Universe: compute per-category % change vs JSON
    universe = raw["universe"]
    json_by_cat_u = {r["category"]: r for r in universe if r["format"] == "json"}
    for r in universe:
        if r["format"] == "json":
            continue
        cat = r["category"]
        jref = json_by_cat_u.get(cat)
        if not jref:
            continue
        tok_pct = (r["total_api_tokens"] - jref["total_api_tokens"]) / jref["total_api_tokens"] * 100
        acc_pct = (r["pass_rate"] - jref["pass_rate"]) / jref["pass_rate"] * 100 if jref["pass_rate"] > 0 else (
            0 if r["pass_rate"] == 0 else 100)
        points.append({
            "benchmark": "MCP-Universe",
            "category": cat,
            "format": r["format"],
            "token_pct": tok_pct,
            "acc_pct": acc_pct,
            "acc_abs": r["pass_rate"],
            "num_tasks": r["num_tasks"],
        })

    return points, raw


def fig1_overview(points):
    """Overall tradeoff scatter with category range whiskers."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Aggregate per (format, benchmark)
    groups = {}
    for p in points:
        key = (p["format"], p["benchmark"])
        groups.setdefault(key, []).append(p)

    for (fmt, bench), pts in groups.items():
        # Weighted average
        total_tasks = sum(p["num_tasks"] for p in pts)
        avg_tok = sum(p["token_pct"] * p["num_tasks"] for p in pts) / total_tasks
        avg_acc = sum(p["acc_pct"] * p["num_tasks"] for p in pts) / total_tasks

        # Min-max ranges
        tok_min = min(p["token_pct"] for p in pts)
        tok_max = max(p["token_pct"] for p in pts)
        acc_min = min(p["acc_pct"] for p in pts)
        acc_max = max(p["acc_pct"] for p in pts)

        label = f"{FORMAT_LABELS[fmt]} ({BENCH_LABELS[bench]})"
        ax.errorbar(avg_tok, avg_acc,
                    xerr=[[avg_tok - tok_min], [tok_max - avg_tok]],
                    yerr=[[avg_acc - acc_min], [acc_max - avg_acc]],
                    fmt=MARKERS[bench], color=COLORS[fmt], markersize=10,
                    capsize=4, capthick=1.5, linewidth=1.5,
                    label=label, zorder=5)

    # Reference lines
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Quadrant annotations
    ax.text(0.02, 0.98, 'More tokens,\nhigher accuracy', transform=ax.transAxes,
            fontsize=7, alpha=0.3, va='top', ha='left')
    ax.text(0.98, 0.98, 'Fewer tokens,\nhigher accuracy', transform=ax.transAxes,
            fontsize=7, alpha=0.3, va='top', ha='right')
    ax.text(0.02, 0.02, 'More tokens,\nlower accuracy', transform=ax.transAxes,
            fontsize=7, alpha=0.3, va='bottom', ha='left')
    ax.text(0.98, 0.02, 'Fewer tokens,\nlower accuracy', transform=ax.transAxes,
            fontsize=7, alpha=0.3, va='bottom', ha='right')

    # JSON baseline marker
    ax.plot(0, 0, 's', color='#2ECC71', markersize=12, zorder=6, label='JSON (baseline)')

    ax.set_xlabel('Token Change vs JSON (%)', fontsize=11)
    ax.set_ylabel('Accuracy Change vs JSON (%)', fontsize=11)
    ax.legend(fontsize=8, loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tradeoff_overview.pdf", bbox_inches='tight', dpi=300)
    print(f"Saved {OUT_DIR / 'tradeoff_overview.pdf'}")
    plt.close(fig)


def fig2_categories(points):
    """Per-category scatter plot."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for p in points:
        ax.scatter(p["token_pct"], p["acc_pct"],
                   c=COLORS[p["format"]], marker=MARKERS[p["benchmark"]],
                   s=80, alpha=0.7, edgecolors='white', linewidth=0.5, zorder=4)

    # Add category labels using adjustText to avoid overlap
    from adjustText import adjust_text
    texts = []
    for p in points:
        cat_short = p["category"].replace("_", " ").replace("automation", "auto.")
        cat_short = cat_short.replace("management", "mgmt").replace("navigation", "nav.")
        cat_short = cat_short.replace("analysis", "anal.").replace("financial", "fin.")
        cat_short = cat_short.replace("repository", "repo").replace("location", "loc.")
        cat_short = cat_short.replace("file system", "file_sys")
        texts.append(ax.text(p["token_pct"], p["acc_pct"], cat_short,
                             fontsize=6, alpha=0.6))
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3, lw=0.5))

    # Reference lines
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # JSON baseline
    ax.plot(0, 0, 's', color='#2ECC71', markersize=12, zorder=6)

    # Legend entries
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["toon"],
               markersize=9, label='TOON (BenchPP)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS["toon"],
               markersize=9, label='TOON (Universe)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["tron"],
               markersize=9, label='TRON (BenchPP)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS["tron"],
               markersize=9, label='TRON (Universe)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ECC71',
               markersize=9, label='JSON (baseline)'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower left', framealpha=0.9)

    ax.set_xlabel('Token Change vs JSON (%)', fontsize=11)
    ax.set_ylabel('Accuracy Change vs JSON (%)', fontsize=11)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tradeoff_categories.pdf", bbox_inches='tight', dpi=300)
    print(f"Saved {OUT_DIR / 'tradeoff_categories.pdf'}")
    plt.close(fig)


def fig3_spider(raw):
    """Spider/radar chart — one panel per benchmark."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), subplot_kw=dict(polar=True))

    benchmarks = [
        ("MCPToolBenchPP", raw["benchpp"], "pass_at_1"),
        ("MCP-Universe", raw["universe"], "pass_rate"),
    ]

    fmt_colors = {"json": "#2ECC71", "toon": "#E74C3C", "tron": "#2980B9"}
    fmt_labels = {"json": "JSON", "toon": "TOON", "tron": "TRON"}

    for ax, (bench_name, results, metric_key) in zip(axes, benchmarks):
        categories = sorted(set(r["category"] for r in results))
        cat_labels = [c.replace("_", "\n") for c in categories]
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # close polygon

        for fmt in ["json", "toon", "tron"]:
            values = []
            for cat in categories:
                entries = [r for r in results if r["category"] == cat and r["format"] == fmt]
                if entries:
                    values.append(entries[0][metric_key])
                else:
                    values.append(0)
            values += values[:1]  # close polygon

            ax.plot(angles, values, 'o-', color=fmt_colors[fmt], linewidth=1.5,
                    markersize=4, label=fmt_labels[fmt], alpha=0.8)
            ax.fill(angles, values, color=fmt_colors[fmt], alpha=0.08)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cat_labels, fontsize=7)
        ax.set_title(bench_name, fontsize=11, fontweight='bold', pad=20)
        ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.3, 1.1))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "category_spider.pdf", bbox_inches='tight', dpi=300)
    print(f"Saved {OUT_DIR / 'category_spider.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    points, raw = load_data()
    fig1_overview(points)
    fig2_categories(points)
    fig3_spider(raw)
    print("All figures generated.")
