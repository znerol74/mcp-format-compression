"""Presentation figures for the status slide deck.

Generates three PNG + PDF figures:
  1. hero_qwen30b.(pdf|png)   — qwen3-30b on StableToolBench triptych (SoPR, tokens, turns)
  2. accuracy_matrix.(pdf|png) — grouped bar chart, accuracy by benchmark × model × format
  3. tradeoff_scatter.(pdf|png) — token Δ vs accuracy Δ scatter, one dot per (model, benchmark, format)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent

FMT_COLORS = {"json": "#7F8C8D", "toon": "#2ECC71", "tron": "#2980B9"}
FMT_LABELS = {"json": "JSON", "toon": "TOON", "tron": "TRON"}

# Display names shown on plots — includes quantization + MoE active-param info
MODEL_DISPLAY = {
    "qwen3-30b":         "Qwen3-30B",
    "qwen3-32b":         "Qwen3-32B",
    "qwen3-32b-fp16":    "Qwen3-32B (fp16)",
    "qwen3-32b-awq":     "Qwen3-32B (AWQ)",
    "qwen35":            "Qwen3-235B-A22B",
    "mistral-small-24b": "Mistral-Small-24B",
}

# ---- Only validated results are included. ----
# Excluded (and why):
#   StableToolBench × mistral-small-24b — TRON inference produced empty files; SoPR judge pending
#   StableToolBench × deepseek-r1-32b   — SoPR judge pending
#   MCP-Universe   × qwen3-30b          — TRON ran zero tokens on 3/5 categories
#   BFCL           × qwen3-32b           — exp2 runs returned zero token data (broken)
# (benchmark, model, fmt) -> (accuracy_pct, tokens, turns)
DATA = {
    # StableToolBench (SoPR %)
    ("StableToolBench", "qwen3-30b", "json"):         (33.3, 5_084_707, 6.4),
    ("StableToolBench", "qwen3-30b", "toon"):         (59.4, 3_261_797, 3.2),
    ("StableToolBench", "qwen3-30b", "tron"):         (57.0, 3_266_468, 3.4),
    ("StableToolBench", "qwen3-32b-fp16", "json"):    (31.2, 2_248_975, 3.5),
    ("StableToolBench", "qwen3-32b-fp16", "toon"):    (31.5, 3_223_430, 3.4),
    ("StableToolBench", "qwen3-32b-fp16", "tron"):    (34.8, 3_176_990, 3.4),
    ("StableToolBench", "qwen3-32b-awq",  "json"):    (34.5, 2_192_068, 3.4),
    ("StableToolBench", "qwen3-32b-awq",  "toon"):    (32.1, 3_246_290, 3.4),
    ("StableToolBench", "qwen3-32b-awq",  "tron"):    (33.0, 3_119_773, 3.4),
    ("StableToolBench", "qwen35",         "json"):    (33.3, 2_493_429, 3.5),
    ("StableToolBench", "qwen35",         "toon"):    (31.8, 3_801_077, 3.5),
    ("StableToolBench", "qwen35",         "tron"):    (23.6, 1_823_294, 2.4),

    # MCPToolBenchPP (pass@1 %)
    ("MCPToolBenchPP", "qwen3-30b", "json"): (54.8, 32_266_835, 1.0),
    ("MCPToolBenchPP", "qwen3-30b", "toon"): (36.0, 23_931_262, 1.0),
    ("MCPToolBenchPP", "qwen3-30b", "tron"): (49.9, 26_657_364, 1.0),
    ("MCPToolBenchPP", "qwen3-32b", "json"): (33.9, 23_364_847, 1.0),
    ("MCPToolBenchPP", "qwen3-32b", "toon"): (20.3, 16_191_564, 1.0),
    ("MCPToolBenchPP", "qwen3-32b", "tron"): (32.5, 19_692_443, 1.0),

    # MCP-Universe (pass rate %) — qwen3-32b only (30b TRON run incomplete)
    ("MCP-Universe", "qwen3-32b", "json"): (13.1, 36_829_866, 10.4),
    ("MCP-Universe", "qwen3-32b", "toon"): (4.0,  39_479_610, 14.2),
    ("MCP-Universe", "qwen3-32b", "tron"): (12.5, 25_068_509, 8.4),

    # BFCL non-live (AST %) — qwen3-30b and mistral-small-24b only (32b exp2 broken)
    ("BFCL", "qwen3-30b", "json"): (95.0,   759_343, 1.0),
    ("BFCL", "qwen3-30b", "toon"): (55.0,   783_365, 1.0),
    ("BFCL", "qwen3-30b", "tron"): (94.4,   756_135, 1.0),
    ("BFCL", "mistral-small-24b", "json"): (93.6, 761_169, 1.0),
    ("BFCL", "mistral-small-24b", "toon"): (51.2, 796_022, 1.0),
    ("BFCL", "mistral-small-24b", "tron"): (93.1, 784_939, 1.0),
}

ACC_LABEL = {
    "StableToolBench": "SoPR (%)",
    "MCPToolBenchPP":  "pass@1 (%)",
    "MCP-Universe":    "pass rate (%)",
    "BFCL":            "AST match (%)",
}


def hero_qwen30b():
    """Three-panel hero chart: qwen3-30b on StableToolBench."""
    fmts = ["json", "toon", "tron"]
    accs  = [DATA[("StableToolBench", "qwen3-30b", f)][0] for f in fmts]
    toks  = [DATA[("StableToolBench", "qwen3-30b", f)][1] / 1e6 for f in fmts]
    turns = [DATA[("StableToolBench", "qwen3-30b", f)][2] for f in fmts]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = [FMT_COLORS[f] for f in fmts]
    labels = [FMT_LABELS[f] for f in fmts]

    for ax, values, title, ylabel, fmt_str in zip(
        axes,
        [accs, toks, turns],
        ["Accuracy (SoPR)", "Total tokens", "Turns per task"],
        ["%", "M tokens", "avg turns"],
        ["{:.1f}%", "{:.2f}M", "{:.1f}"],
    ):
        bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.set_axisbelow(True)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, v, fmt_str.format(v),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.18)

    fig.suptitle(f"{MODEL_DISPLAY['qwen3-30b']} on StableToolBench — format choice is a free accuracy win",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "hero_qwen30b.pdf", bbox_inches="tight")
    fig.savefig(OUT / "hero_qwen30b.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def accuracy_matrix():
    """Grouped bar chart: accuracy for every (benchmark, model) with three format bars."""
    rows = [
        ("StableToolBench", "qwen3-30b"),
        ("StableToolBench", "qwen3-32b-fp16"),
        ("StableToolBench", "qwen3-32b-awq"),
        ("StableToolBench", "qwen35"),
        ("MCPToolBenchPP",  "qwen3-30b"),
        ("MCPToolBenchPP",  "qwen3-32b"),
        ("MCP-Universe",    "qwen3-32b"),
        ("BFCL",            "qwen3-30b"),
        ("BFCL",            "mistral-small-24b"),
    ]
    labels = [f"{b}\n{MODEL_DISPLAY[m]}" for b, m in rows]
    x = np.arange(len(rows))
    width = 0.27

    fig, ax = plt.subplots(figsize=(13, 5.2))
    for i, fmt in enumerate(["json", "toon", "tron"]):
        vals = [DATA[(b, m, fmt)][0] for b, m in rows]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=FMT_LABELS[fmt], color=FMT_COLORS[fmt],
                      edgecolor="black", linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy (benchmark-specific metric, %)")
    ax.set_title("Accuracy by benchmark × model × format", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_matrix.pdf", bbox_inches="tight")
    fig.savefig(OUT / "accuracy_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def tradeoff_scatter():
    """Token Δ vs accuracy Δ scatter, one point per (model, benchmark, fmt ∈ {toon, tron})."""
    # Build points
    fmts = ["toon", "tron"]
    bench_markers = {
        "StableToolBench": "o",
        "MCPToolBenchPP":  "s",
        "MCP-Universe":    "^",
        "BFCL":            "D",
    }

    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.axvline(0, color="black", lw=0.6, alpha=0.5)
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    # quadrant shading
    ax.axhspan(0, 40, xmin=0, xmax=0.5, color="#2ECC71", alpha=0.05)

    seen_models = set()
    for (bench, model, fmt), (acc, tok, turns) in DATA.items():
        if fmt == "json" or "pending" in str(acc):
            continue
        ref = DATA.get((bench, model, "json"))
        if ref is None:
            continue
        acc_delta = acc - ref[0]
        tok_delta = (tok - ref[1]) / ref[1] * 100
        ax.scatter(tok_delta, acc_delta,
                   marker=bench_markers[bench],
                   s=110, c=FMT_COLORS[fmt], edgecolors="black", linewidths=0.7,
                   alpha=0.9)
        short = MODEL_DISPLAY.get(model, model)
        ax.annotate(short, (tok_delta, acc_delta), xytext=(6, 4),
                    textcoords="offset points", fontsize=7.5, alpha=0.8)

    ax.set_xlabel("Token change vs JSON (%)", fontsize=10)
    ax.set_ylabel("Accuracy change vs JSON (pp)", fontsize=10)
    ax.set_title("Token–accuracy tradeoff per format (vs JSON baseline)",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)

    # Legends
    fmt_handles = [plt.Line2D([0], [0], marker="o", color="w", label=FMT_LABELS[f],
                              markerfacecolor=FMT_COLORS[f], markeredgecolor="black",
                              markersize=11) for f in fmts]
    bench_handles = [plt.Line2D([0], [0], marker=m, color="w", label=b,
                                markerfacecolor="#888", markeredgecolor="black",
                                markersize=10) for b, m in bench_markers.items()]
    leg1 = ax.legend(handles=fmt_handles, loc="lower left", title="Format", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=bench_handles, loc="lower right", title="Benchmark", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT / "tradeoff_scatter.pdf", bbox_inches="tight")
    fig.savefig(OUT / "tradeoff_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def absolute_scatter():
    """Single scatter: X = total tokens (log), Y = accuracy %. Kept for compact use.
    Every validated run is one point. Color = format, marker = benchmark."""
    bench_markers = {
        "StableToolBench": "o",
        "MCPToolBenchPP":  "s",
        "MCP-Universe":    "^",
        "BFCL":            "D",
    }
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for (bench, model, fmt), (acc, tok, _turns) in DATA.items():
        ax.scatter(tok, acc, marker=bench_markers[bench], s=120,
                   c=FMT_COLORS[fmt], edgecolors="black", linewidths=0.7,
                   alpha=0.9, zorder=3)
        short = MODEL_DISPLAY.get(model, model)
        ax.annotate(short, (tok, acc), xytext=(6, 4),
                    textcoords="offset points", fontsize=7.5, alpha=0.85)
    pairs = {}
    for (bench, model, fmt), (acc, tok, _) in DATA.items():
        pairs.setdefault((bench, model), {})[fmt] = (tok, acc)
    for (bench, model), fmts in pairs.items():
        if "json" not in fmts: continue
        jx, jy = fmts["json"]
        for other in ("toon", "tron"):
            if other in fmts:
                ox, oy = fmts[other]
                ax.plot([jx, ox], [jy, oy],
                        color=FMT_COLORS[other], lw=0.8, alpha=0.35, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("Total tokens (log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs tokens — every validated run", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100); ax.set_xlim(5e5, 5e7)
    ax.grid(True, which="both", alpha=0.3, linestyle=":"); ax.set_axisbelow(True)
    fmt_handles = [plt.Line2D([0], [0], marker="o", color="w", label=FMT_LABELS[f],
                              markerfacecolor=FMT_COLORS[f], markeredgecolor="black",
                              markersize=11) for f in ("json", "toon", "tron")]
    bench_handles = [plt.Line2D([0], [0], marker=m, color="w", label=b,
                                markerfacecolor="#888", markeredgecolor="black",
                                markersize=10) for b, m in bench_markers.items()]
    leg1 = ax.legend(handles=fmt_handles, loc="lower left", title="Format", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=bench_handles, loc="lower right", title="Benchmark", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "absolute_scatter.pdf", bbox_inches="tight")
    fig.savefig(OUT / "absolute_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def faceted_scatter():
    """2x2 faceted scatter — one panel per benchmark. Far easier to read.
    Each panel uses a linear x-axis tuned to its own range so labels have breathing room.
    JSON → compressed arrows show the move induced by each format."""
    bench_order = ["StableToolBench", "MCPToolBenchPP", "MCP-Universe", "BFCL"]
    acc_labels = {
        "StableToolBench": "SoPR (%)",
        "MCPToolBenchPP":  "pass@1 (%)",
        "MCP-Universe":    "pass rate (%)",
        "BFCL":            "AST match (%)",
    }
    # tokens are shown in millions for readability
    x_units = {
        "StableToolBench": 1e6,
        "MCPToolBenchPP":  1e6,
        "MCP-Universe":    1e6,
        "BFCL":            1e3,   # kilotokens
    }
    x_unit_label = {
        "StableToolBench": "Total tokens (millions)",
        "MCPToolBenchPP":  "Total tokens (millions)",
        "MCP-Universe":    "Total tokens (millions)",
        "BFCL":            "Total tokens (thousands)",
    }

    # Manual label offsets for crowded clusters (benchmark, model) -> (dx_pts, dy_pts, ha)
    label_offsets = {
        ("StableToolBench", "qwen3-30b"):        (12,   0, "left"),
        ("StableToolBench", "qwen3-32b-fp16"):   (-12, -22, "right"),
        ("StableToolBench", "qwen3-32b-awq"):    (-12,  22, "right"),
        ("StableToolBench", "qwen35"):           (12,   22, "left"),
        ("MCPToolBenchPP",  "qwen3-30b"):        (-14, 14, "right"),
        ("MCPToolBenchPP",  "qwen3-32b"):        (-14, 14, "right"),
        ("MCP-Universe",    "qwen3-32b"):        (-14, 14, "right"),
        ("BFCL",            "qwen3-30b"):        (12,  -20, "left"),
        ("BFCL",            "mistral-small-24b"):(12,  14, "left"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for ax, bench in zip(axes, bench_order):
        unit = x_units[bench]
        by_model = {}
        for (b, m, f), (acc, tok, _) in DATA.items():
            if b != bench: continue
            by_model.setdefault(m, {})[f] = (tok / unit, acc)

        # arrows first
        for model, fmts in by_model.items():
            if "json" not in fmts: continue
            jx, jy = fmts["json"]
            for other in ("toon", "tron"):
                if other in fmts:
                    ox, oy = fmts[other]
                    ax.annotate("", xy=(ox, oy), xytext=(jx, jy),
                                arrowprops=dict(arrowstyle="->",
                                                color=FMT_COLORS[other],
                                                lw=1.8, alpha=0.6),
                                zorder=2)

        # points
        for model, fmts in by_model.items():
            for fmt, (tok, acc) in fmts.items():
                ax.scatter(tok, acc, s=240, c=FMT_COLORS[fmt],
                           edgecolors="black", linewidths=1.2, zorder=3)
            if "json" in fmts:
                jx, jy = fmts["json"]
                full_name = MODEL_DISPLAY.get(model, model)
                dx, dy, ha = label_offsets.get((bench, model), (10, 10, "left"))
                ax.annotate(full_name, (jx, jy), xytext=(dx, dy),
                            textcoords="offset points", fontsize=11,
                            fontweight="bold", ha=ha,
                            bbox=dict(boxstyle="round,pad=0.25",
                                      facecolor="white", edgecolor="#888",
                                      alpha=0.85, lw=0.5))

        ax.set_title(bench, fontsize=14, fontweight="bold")
        ax.set_xlabel(x_unit_label[bench], fontsize=11)
        ax.set_ylabel(acc_labels[bench], fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10)

    # global legend
    fmt_handles = [plt.Line2D([0], [0], marker="o", color="w", label=FMT_LABELS[f],
                              markerfacecolor=FMT_COLORS[f], markeredgecolor="black",
                              markersize=14) for f in ("json", "toon", "tron")]
    fig.legend(handles=fmt_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.01), fontsize=13, title="Format",
               title_fontsize=13, framealpha=0.9)
    fig.suptitle("Accuracy vs total tokens",
                 fontsize=15, fontweight="bold", y=1.00)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(OUT / "faceted_scatter.pdf", bbox_inches="tight")
    fig.savefig(OUT / "faceted_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    hero_qwen30b()
    accuracy_matrix()
    tradeoff_scatter()
    absolute_scatter()
    faceted_scatter()
    for p in sorted(OUT.glob("*.p*")):
        print(p)
