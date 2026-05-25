#!/usr/bin/env python3
"""Generate model-format tradeoff figure — per-benchmark data points with real data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# Format colors (matching existing figures)
FMT_COLORS = {"TOON": "#2ECC71", "TRON": "#2980B9"}

# Benchmark markers
BENCH_MARKERS = {"STB": "o", "BPP": "s", "UNI": "^", "BFCL": "D"}
BENCH_LABELS = {"STB": "StableToolBench", "BPP": "MCPToolBenchPP", "UNI": "MCP-Universe", "BFCL": "BFCL"}

# Real data: each point = (token_change%, accuracy_change_pp)
# model -> format -> [(bench, tok%, acc_pp), ...]
real_data = {
    "Qwen3-30B": {
        "TOON": [
            ("STB", -35.9, +19.7),
            ("BPP", -8.8, -18.8),
            ("BFCL", +3.2, -40.0),
        ],
        "TRON": [
            ("STB", -35.8, +25.5),
            ("BPP", -20.9, -4.9),
            ("BFCL", -0.4, -0.6),
        ],
    },
    "Qwen3.5-35B": {
        "TOON": [
            ("STB", +52.4, +0.0),
        ],
        "TRON": [
            ("STB", -26.9, -7.6),
        ],
    },
    "Qwen3-32B": {
        "TOON": [
            ("UNI", +7.2, -9.1),
        ],
        "TRON": [
            ("UNI", -31.9, -0.6),
        ],
    },
}

# Model sizes for marker sizing (bigger model = bigger marker)
MODEL_PARAM_SIZE = {
    "Qwen3-30B": 30,
    "Qwen3-32B": 32,
    "Qwen3.5-35B": 35,
}

def param_to_markersize(params):
    return 6 + (params / 10)  # 30B -> 9, 35B -> 9.5, 120B -> 18


fig, ax = plt.subplots(figsize=(11, 7.5))

# Plot each data point
for model, formats in real_data.items():
    ms = param_to_markersize(MODEL_PARAM_SIZE[model])
    for fmt, points in formats.items():
        for bench, tok, acc in points:
            ax.plot(tok, acc,
                    marker=BENCH_MARKERS[bench],
                    color=FMT_COLORS[fmt],
                    markersize=ms,
                    markeredgecolor='white',
                    markeredgewidth=0.8,
                    zorder=5)
            # Label with model name (small, offset)
            offset = (6, 6) if acc > 0 else (6, -12)
            ax.annotate(f"{model}\n({BENCH_LABELS[bench]})",
                       (tok, acc), textcoords="offset points",
                       xytext=offset, fontsize=6, color='gray', alpha=0.8)

# JSON baseline at origin
ax.plot(0, 0, '*', color='#E74C3C', markersize=18, zorder=6,
        markeredgecolor='black', markeredgewidth=0.5)
ax.annotate("JSON\n(baseline)", (0, 0), textcoords="offset points",
            xytext=(12, -18), fontsize=9, color='#E74C3C', fontweight='bold')

# Quadrant shading
ax.fill_between([-60, 0], 0, 35, color='#d4edda', alpha=0.12, zorder=0)
ax.fill_between([0, 60], -45, 0, color='#f8d7da', alpha=0.12, zorder=0)

ax.text(-50, 30, "Ideal: fewer tokens,\nhigher accuracy",
        fontsize=8.5, color='#155724', fontstyle='italic', ha='center', va='center')
ax.text(40, -38, "Worst: more tokens,\nlower accuracy",
        fontsize=8.5, color='#721c24', fontstyle='italic', ha='center', va='center')

# Reference lines
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

# Legend: formats (colors)
fmt_handles = [
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#E74C3C',
               markeredgecolor='black', markeredgewidth=0.5, markersize=12, label='JSON (baseline)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ECC71',
               markersize=10, label='TOON'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2980B9',
               markersize=10, label='TRON'),
]

# Legend: benchmarks (shapes)
bench_handles = [
    plt.Line2D([0], [0], marker=m, color='w', markerfacecolor='gray',
               markeredgecolor='white', markeredgewidth=0.5,
               markersize=9, label=BENCH_LABELS[b])
    for b, m in BENCH_MARKERS.items() if b in ["STB", "BPP", "UNI", "BFCL"]
]

legend1 = ax.legend(handles=fmt_handles, title="Format", loc='lower left',
                    fontsize=9, title_fontsize=10, framealpha=0.9)
ax.add_artist(legend1)
ax.legend(handles=bench_handles, title="Benchmark", loc='upper right',
          fontsize=9, title_fontsize=10, framealpha=0.9)

ax.set_xlabel("Token Change vs JSON (%)", fontsize=12, labelpad=8)
ax.set_ylabel("Accuracy Change vs JSON (pp)", fontsize=12, labelpad=8)
ax.set_title("Token-Accuracy Tradeoff by Format, Model, and Benchmark",
             fontsize=13, fontweight='bold')

ax.set_xlim(-55, 60)
ax.set_ylim(-45, 35)
ax.grid(True, alpha=0.15)
ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(OUT_DIR / "model_format_tradeoff.pdf", dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR / "model_format_tradeoff.png", dpi=150, bbox_inches='tight')
print("Saved model_format_tradeoff.pdf and .png")