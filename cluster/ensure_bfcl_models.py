#!/usr/bin/env python3
"""Inject 6 BFCL model registry entries per cluster model into model_config.py.

Idempotent: skips entries that already exist. Mirrors the qwen3-30b-local pattern:
  <tag>            : baseline (json input, python output)
  <tag>-toon       : exp1 TOON input, python output
  <tag>-tron       : exp1 TRON input, python output
  <tag>-json-full  : exp2 JSON input + JSON output
  <tag>-toon-full  : exp2 TOON input + TOON output
  <tag>-tron-full  : exp2 TRON input + TRON output

Usage:
  python cluster/ensure_bfcl_models.py <TAG> <HF_MODEL_NAME> [--display "Display Name"] [--org Org]
"""
import argparse
import re
import sys
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "constants" / "model_config.py"

VARIANTS = [
    ("",            "Baseline"),
    ("-toon",       "TOON"),
    ("-tron",       "TRON"),
    ("-json-full",  "JSON Full"),
    ("-toon-full",  "TOON Full"),
    ("-tron-full",  "TRON Full"),
]

ENTRY_TEMPLATE = '''    "{key}": ModelConfig(
        model_name="{hf}",
        display_name="{display} {variant_label} (Prompt)",
        url="https://huggingface.co/{hf}",
        org="{org}",
        license="Apache 2.0",
        model_handler=OpenWebUICompletionsHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    ),
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="Model tag, e.g. mistral-small-24b")
    ap.add_argument("hf", help="HuggingFace model id, e.g. stelterlab/Mistral-Small-24B-Instruct-2501-AWQ")
    ap.add_argument("--display", default=None, help="Pretty display name; defaults to tag")
    ap.add_argument("--org", default="Cluster", help="Org label")
    args = ap.parse_args()

    display = args.display or args.tag
    text = CONFIG.read_text()

    # Anchor: match the line `}` immediately before the blank line + `# Inference through local hosting`
    # comment. Use a specific regex so we don't match nested `}` inside other ModelConfig entries.
    anchor_re = re.compile(r'(\n\}\n)(\s*\n# Inference through local hosting\b)', re.M)
    m = anchor_re.search(text)
    if not m:
        sys.exit(f"ERROR: could not locate api_inference_model_map closing brace in {CONFIG}")

    insertions = []
    for suffix, label in VARIANTS:
        key = f"{args.tag}{suffix}"
        if re.search(rf'^\s*"{re.escape(key)}":\s*ModelConfig\(', text, re.M):
            print(f"  skip (exists): {key}")
            continue
        entry = ENTRY_TEMPLATE.format(
            key=key, hf=args.hf, display=display,
            variant_label=label, org=args.org,
        )
        insertions.append(entry)
        print(f"  add:           {key}")

    if not insertions:
        print("Nothing to add.")
        return

    # Insert the new entries right before the `}` (inside the dict).
    insert_at = m.start(1) + 1  # position right after the leading \n, before the `}`
    new_text = text[:insert_at] + "".join(insertions) + text[insert_at:]
    CONFIG.write_text(new_text)
    print(f"Wrote {CONFIG} (+{len(insertions)} entries).")


if __name__ == "__main__":
    main()
