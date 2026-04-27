#!/usr/bin/env python3
"""Option A: one point per (model, format); whiskers = benchmark range.

Reads multi_model_data.json produced by collect_multi_model_data.py.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = OUT_DIR / "multi_model_data.json"

FMT_COLORS = {"TOON": "#2ECC71", "TRON": "#2980B9"}
# Offset TOON/TRON slightly so overlapping whiskers stay readable
FMT_DX = {"TOON": -0.0, "TRON": 0.0}


def mean_min_max(points, key):
    vals = [p[key] for p in points]
    return sum(vals) / len(vals), min(vals), max(vals)


def main():
    data = json.loads(DATA_PATH.read_text())

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    model_order = sorted(data.keys(), key=lambda m: -len(data[m].get("TOON", {})) - len(data[m].get("TRON", {})))

    for model in model_order:
        for fmt in ("TOON", "TRON"):
            bench_data = data[model].get(fmt, {})
            if not bench_data:
                continue
            points = list(bench_data.values())
            tok_mean, tok_min, tok_max = mean_min_max(points, "tok_pct")
            acc_mean, acc_min, acc_max = mean_min_max(points, "acc_pct")

            ax.errorbar(
                tok_mean, acc_mean,
                xerr=[[tok_mean - tok_min], [tok_max - tok_mean]],
                yerr=[[acc_mean - acc_min], [acc_max - acc_mean]],
                fmt="o",
                color=FMT_COLORS[fmt],
                markersize=11,
                markeredgecolor="white",
                markeredgewidth=1.0,
                capsize=4,
                elinewidth=1.2,
                zorder=5,
            )

            # Label: model name next to each point
            offset_y = 10 if fmt == "TOON" else -16
            ax.annotate(
                f"{model}",
                (tok_mean, acc_mean),
                textcoords="offset points",
                xytext=(8, offset_y),
                fontsize=8,
                color=FMT_COLORS[fmt],
                fontweight="bold",
                alpha=0.95,
            )

    # JSON baseline at origin
    ax.plot(0, 0, "*", color="#E74C3C", markersize=18,
            markeredgecolor="black", markeredgewidth=0.5, zorder=6)
    ax.annotate("JSON\n(baseline)", (0, 0), textcoords="offset points",
                xytext=(10, -20), fontsize=9, color="#E74C3C", fontweight="bold")

    # Reference lines
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Quadrant shading
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.fill_between([min(-80, xlim[0]), 0], 0, max(100, ylim[1]),
                    color="#d4edda", alpha=0.12, zorder=0)
    ax.fill_between([0, max(80, xlim[1])], min(-80, ylim[0]), 0,
                    color="#f8d7da", alpha=0.12, zorder=0)

    # Axis labels
    ax.set_xlabel("Token Change vs JSON (%), mean across benchmarks",
                  fontsize=11, labelpad=8)
    ax.set_ylabel("Accuracy Change vs JSON (%), mean across benchmarks",
                  fontsize=11, labelpad=8)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#E74C3C",
               markeredgecolor="black", markeredgewidth=0.5,
               markersize=13, label="JSON (baseline)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FMT_COLORS["TOON"],
               markersize=10, label="TOON"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FMT_COLORS["TRON"],
               markersize=10, label="TRON"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=9,
              framealpha=0.9, edgecolor="lightgray")

    # Corner hints
    ax.text(0.02, 0.98, "Fewer tokens,\nhigher accuracy",
            transform=ax.transAxes, fontsize=8, alpha=0.45,
            va="top", ha="left", fontstyle="italic")
    ax.text(0.98, 0.02, "More tokens,\nlower accuracy",
            transform=ax.transAxes, fontsize=8, alpha=0.45,
            va="bottom", ha="right", fontstyle="italic")

    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out = OUT_DIR / "model_tradeoff.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / "model_tradeoff.png", bbox_inches="tight", dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
