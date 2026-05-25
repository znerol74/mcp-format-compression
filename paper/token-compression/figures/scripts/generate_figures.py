#!/usr/bin/env python3
"""Generate paper figures from experiment results."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = ROOT / "experiment_logs" / "analysis" / "combined_results.json"
OUT_DIR = Path(__file__).resolve().parent

# Colors and markers
COLORS = {"toon": "#2ECC71", "tron": "#2980B9"}
MARKERS = {"MCPToolBenchPP": "o", "MCP-Universe": "^"}
FORMAT_LABELS = {"toon": "TOON", "tron": "TRON"}
BENCH_LABELS = {"MCPToolBenchPP": "BenchPP", "MCP-Universe": "Universe"}

# Tool counts per category (from benchmark data files)
TOOL_COUNTS = {
    # MCPToolBenchPP
    "finance": 1, "search": 5, "pay": 6,
    "file_system": 11, "map": 22, "browser": 32,
    # MCP-Universe
    "financial_analysis": 10, "3d_design": 13,
    "browser_automation": 29, "location_navigation": 32,
    "repository_management": 73,
}

CAT_SHORT = {
    "finance": "Finance", "search": "Search", "pay": "Pay",
    "file_system": "FileSys", "map": "Map", "browser": "Browser",
    "financial_analysis": "Finance", "3d_design": "3D Design",
    "browser_automation": "Browser", "location_navigation": "Location",
    "repository_management": "Repo Mgmt",
}


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

    # Attach tool counts
    for p in points:
        p["num_tools"] = TOOL_COUNTS.get(p["category"], 0)

    return points, raw


def fig1_overview(points):
    """Overall tradeoff scatter with category range whiskers."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Aggregate per (format, benchmark)
    groups = {}
    for p in points:
        key = (p["format"], p["benchmark"])
        groups.setdefault(key, []).append(p)

    # Sort keys for consistent legend order: TOON BenchPP, TOON Universe, TRON BenchPP, TRON Universe
    sorted_keys = sorted(groups.keys(), key=lambda k: (k[0], k[1]))

    for (fmt, bench), pts in sorted(groups.items(), key=lambda k: (k[0][0], k[0][1])):
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

    # Quadrant annotations — left = negative token change = fewer tokens
    ax.text(0.02, 0.98, 'Fewer tokens,\nhigher accuracy', transform=ax.transAxes,
            fontsize=7, alpha=0.3, va='top', ha='left')
    ax.text(0.02, 0.02, 'Fewer tokens,\nlower accuracy', transform=ax.transAxes,
            fontsize=7, alpha=0.3, va='bottom', ha='left')
    ax.text(0.98, 0.02, 'More tokens,\nlower accuracy', transform=ax.transAxes,
            fontsize=7, alpha=0.3, va='bottom', ha='right')

    # JSON baseline marker
    ax.plot(0, 0, 's', color='#E74C3C', markersize=12, zorder=6)

    # Build legend manually to avoid overlap with data
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#E74C3C',
               markersize=9, label='JSON (baseline)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["toon"],
               markersize=9, label='TOON (BenchPP)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS["toon"],
               markersize=9, label='TOON (Universe)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["tron"],
               markersize=9, label='TRON (BenchPP)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS["tron"],
               markersize=9, label='TRON (Universe)'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper right',
              framealpha=0.9, edgecolor='lightgray')

    ax.set_xlabel('Token Change vs JSON (%)', fontsize=11)
    ax.set_ylabel('Accuracy Change vs JSON (%)', fontsize=11)
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
                   s=80, alpha=0.7, zorder=4)

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

    # Quadrant annotations — left = negative token change = fewer tokens
    ax.text(0.02, 0.98, 'Fewer tokens,\nhigher accuracy', transform=ax.transAxes,
            fontsize=7, alpha=0.3, va='top', ha='left')
    ax.text(0.02, 0.02, 'Fewer tokens,\nlower accuracy', transform=ax.transAxes,
            fontsize=7, alpha=0.3, va='bottom', ha='left')
    ax.text(0.98, 0.02, 'More tokens,\nlower accuracy', transform=ax.transAxes,
            fontsize=7, alpha=0.3, va='bottom', ha='right')

    # JSON baseline
    ax.plot(0, 0, 's', color='#E74C3C', markersize=12, zorder=6)

    # Legend entries
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#E74C3C',
               markersize=9, label='JSON (baseline)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["toon"],
               markersize=9, label='TOON (BenchPP)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS["toon"],
               markersize=9, label='TOON (Universe)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["tron"],
               markersize=9, label='TRON (BenchPP)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS["tron"],
               markersize=9, label='TRON (Universe)'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper right',
              framealpha=0.9, edgecolor='lightgray')

    ax.set_xlabel('Token Change vs JSON (%)', fontsize=11)
    ax.set_ylabel('Accuracy Change vs JSON (%)', fontsize=11)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tradeoff_categories.pdf", bbox_inches='tight', dpi=300)
    print(f"Saved {OUT_DIR / 'tradeoff_categories.pdf'}")
    plt.close(fig)


def fig1_overview_wide(points):
    """Wide rectangular version of tradeoff overview for title figure embedding."""
    fig, ax = plt.subplots(figsize=(8, 3.5))

    groups = {}
    for p in points:
        key = (p["format"], p["benchmark"])
        groups.setdefault(key, []).append(p)

    for (fmt, bench), pts in sorted(groups.items(), key=lambda k: (k[0][0], k[0][1])):
        total_tasks = sum(p["num_tasks"] for p in pts)
        avg_tok = sum(p["token_pct"] * p["num_tasks"] for p in pts) / total_tasks
        avg_acc = sum(p["acc_pct"] * p["num_tasks"] for p in pts) / total_tasks

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

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.text(0.02, 0.98, 'Fewer tokens,\nhigher accuracy', transform=ax.transAxes,
            fontsize=7, alpha=0.3, va='top', ha='left')
    ax.text(0.02, 0.02, 'Fewer tokens,\nlower accuracy', transform=ax.transAxes,
            fontsize=7, alpha=0.3, va='bottom', ha='left')
    ax.text(0.98, 0.02, 'More tokens,\nlower accuracy', transform=ax.transAxes,
            fontsize=7, alpha=0.3, va='bottom', ha='right')

    ax.plot(0, 0, 's', color='#E74C3C', markersize=12, zorder=6)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#E74C3C',
               markersize=9, label='JSON (baseline)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["toon"],
               markersize=9, label='TOON (BenchPP)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS["toon"],
               markersize=9, label='TOON (Universe)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["tron"],
               markersize=9, label='TRON (BenchPP)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS["tron"],
               markersize=9, label='TRON (Universe)'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper right',
              framealpha=0.9, edgecolor='lightgray')

    ax.set_xlabel('Token Change vs JSON (%)', fontsize=11)
    ax.set_ylabel('Accuracy Change vs JSON (%)', fontsize=11)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tradeoff_overview_wide.pdf", bbox_inches='tight', dpi=300)
    print(f"Saved {OUT_DIR / 'tradeoff_overview_wide.pdf'}")
    plt.close(fig)


def fig_tools_vs_metrics(points):
    """Figure A: 2x2 grid — token savings and accuracy vs number of tools."""
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex='col')

    benchmarks = [("MCPToolBenchPP", "Single-Turn (MCPToolBenchPP)"),
                  ("MCP-Universe", "Multi-Turn (MCP-Universe)")]

    for col, (bench_key, bench_title) in enumerate(benchmarks):
        ax_tok = axes[0, col]
        ax_acc = axes[1, col]

        for fmt in ["toon", "tron"]:
            pts = [p for p in points if p["benchmark"] == bench_key and p["format"] == fmt and p["num_tools"] > 0]
            if not pts:
                continue
            pts.sort(key=lambda p: p["num_tools"])
            x = [p["num_tools"] for p in pts]
            y_tok = [p["token_pct"] for p in pts]
            y_acc = [p["acc_pct"] for p in pts]

            ax_tok.plot(x, y_tok, f'-{MARKERS[bench_key]}', color=COLORS[fmt],
                       markersize=8, linewidth=1.5, label=FORMAT_LABELS[fmt], alpha=0.85)
            ax_acc.plot(x, y_acc, f'-{MARKERS[bench_key]}', color=COLORS[fmt],
                       markersize=8, linewidth=1.5, label=FORMAT_LABELS[fmt], alpha=0.85)

            # Category labels
            for p in pts:
                label = CAT_SHORT.get(p["category"], p["category"])
                ax_tok.annotate(label, (p["num_tools"], p["token_pct"]),
                               textcoords="offset points", xytext=(5, 5),
                               fontsize=6, alpha=0.6)
                ax_acc.annotate(label, (p["num_tools"], p["acc_pct"]),
                               textcoords="offset points", xytext=(5, 5),
                               fontsize=6, alpha=0.6)

        ax_tok.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_acc.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_tok.set_title(bench_title, fontsize=10, fontweight='bold')
        ax_tok.set_ylabel('Token Change vs JSON (%)', fontsize=9)
        ax_acc.set_ylabel('Accuracy Change vs JSON (%)', fontsize=9)
        ax_acc.set_xlabel('Number of Tools', fontsize=9)
        ax_tok.grid(True, alpha=0.2)
        ax_acc.grid(True, alpha=0.2)
        ax_tok.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "tools_vs_metrics.pdf", bbox_inches='tight', dpi=300)
    print(f"Saved {OUT_DIR / 'tools_vs_metrics.pdf'}")
    plt.close(fig)


def fig_tools_scatter(points):
    """Figure B: Scatter colored by tool count."""
    import matplotlib.cm as cm
    import numpy as np
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(7, 5.5))

    pts_with_tools = [p for p in points if p["num_tools"] > 0]
    all_tools = [p["num_tools"] for p in pts_with_tools]
    norm = plt.Normalize(vmin=min(all_tools), vmax=max(all_tools))
    cmap = cm.viridis

    for p in pts_with_tools:
        face_color = cmap(norm(p["num_tools"]))
        edge_color = COLORS[p["format"]]
        marker = MARKERS[p["benchmark"]]
        ax.scatter(p["token_pct"], p["acc_pct"],
                  c=[face_color], edgecolors=edge_color, linewidths=2,
                  marker=marker, s=120, alpha=0.85, zorder=4)

        label = CAT_SHORT.get(p["category"], p["category"])
        ax.annotate(label, (p["token_pct"], p["acc_pct"]),
                   textcoords="offset points", xytext=(6, 4),
                   fontsize=5.5, alpha=0.55)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.plot(0, 0, 's', color='#E74C3C', markersize=12, zorder=6)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Number of Tools', fontsize=9)

    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#E74C3C',
               markersize=9, label='JSON (baseline)'),
        Line2D([0], [0], marker='o', color='w', markeredgecolor=COLORS["toon"],
               markeredgewidth=2, markersize=9, label='TOON (BenchPP)'),
        Line2D([0], [0], marker='^', color='w', markeredgecolor=COLORS["toon"],
               markeredgewidth=2, markersize=9, label='TOON (Universe)'),
        Line2D([0], [0], marker='o', color='w', markeredgecolor=COLORS["tron"],
               markeredgewidth=2, markersize=9, label='TRON (BenchPP)'),
        Line2D([0], [0], marker='^', color='w', markeredgecolor=COLORS["tron"],
               markeredgewidth=2, markersize=9, label='TRON (Universe)'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='upper right',
              framealpha=0.9, edgecolor='lightgray')

    ax.set_xlabel('Token Change vs JSON (%)', fontsize=11)
    ax.set_ylabel('Accuracy Change vs JSON (%)', fontsize=11)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tools_scatter.pdf", bbox_inches='tight', dpi=300)
    print(f"Saved {OUT_DIR / 'tools_scatter.pdf'}")
    plt.close(fig)


def fig_tools_heatmap(points):
    """Figure C: Heatmap of token/accuracy change ordered by tool count."""
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [6, 5]})

    benchmarks = [("MCPToolBenchPP", "MCPToolBenchPP (Single-Turn)"),
                  ("MCP-Universe", "MCP-Universe (Multi-Turn)")]

    for idx, (bench_key, bench_title) in enumerate(benchmarks):
        ax = axes[idx]

        # Get categories for this benchmark, sorted by tool count
        cats = sorted(
            set(p["category"] for p in points if p["benchmark"] == bench_key and p["num_tools"] > 0),
            key=lambda c: TOOL_COUNTS.get(c, 0)
        )

        if not cats:
            continue

        row_labels = [f"{CAT_SHORT.get(c, c)} ({TOOL_COUNTS.get(c, '?')})" for c in cats]
        col_labels = ["TOON\nTokens", "TOON\nAccuracy", "TRON\nTokens", "TRON\nAccuracy"]

        data = np.zeros((len(cats), 4))
        for i, cat in enumerate(cats):
            for p in points:
                if p["benchmark"] != bench_key or p["category"] != cat:
                    continue
                if p["format"] == "toon":
                    data[i, 0] = p["token_pct"]
                    data[i, 1] = p["acc_pct"]
                elif p["format"] == "tron":
                    data[i, 2] = p["token_pct"]
                    data[i, 3] = p["acc_pct"]

        vmax = max(abs(data.min()), abs(data.max()), 1)
        im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto',
                       vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(4))
        ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title(bench_title, fontsize=10, fontweight='bold', pad=10)

        # Annotate cells
        for i in range(len(cats)):
            for j in range(4):
                val = data[i, j]
                color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                       fontsize=7, color=color, fontweight='bold')

    fig.tight_layout(w_pad=3)
    fig.savefig(OUT_DIR / "tools_heatmap.pdf", bbox_inches='tight', dpi=300)
    print(f"Saved {OUT_DIR / 'tools_heatmap.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    points, raw = load_data()
    fig1_overview(points)
    fig1_overview_wide(points)
    fig2_categories(points)
    fig_tools_vs_metrics(points)
    fig_tools_scatter(points)
    fig_tools_heatmap(points)
    print("All figures generated.")
