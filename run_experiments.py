#!/usr/bin/env python3
"""
Unified experiment runner for cross-benchmark MCP format compression paper.

Orchestrates experiments across all three benchmarks:
  - MCPToolBenchPP  (single-turn tool calling)
  - MCP-Universe    (multi-turn ReAct agent)
  - mcp-bench       (multi-round planner)

Experiments:
  exp0  - Offline token analysis (no API calls)
  exp1  - Input compression only (schema+results compressed, tool calls JSON)
  exp2  - Full compression (schema+results+tool calls compressed)
  exp3  - Model comparison (multiple models, full compression)

Usage:
  python run_experiments.py --exp exp1 exp2
  python run_experiments.py --exp exp0          # token analysis only
  python run_experiments.py --exp exp3 --local  # include local LLM (23:00-06:00)
"""

import argparse
import subprocess
import os
import sys
import time
import json
import yaml
import copy
from datetime import datetime, timedelta
from pathlib import Path

# ── Project paths ───────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
BENCHPP_DIR = ROOT / "MCPToolBenchPP"
UNIVERSE_DIR = ROOT / "MCP-Universe"
MCPBENCH_DIR = ROOT / "mcp-bench"

BENCHPP_PYTHON = BENCHPP_DIR / ".venv" / "bin" / "python3"
UNIVERSE_PYTHON = UNIVERSE_DIR / ".venv" / "bin" / "python3"
MCPBENCH_PYTHON = MCPBENCH_DIR / "venv" / "bin" / "python3"

LOG_DIR = ROOT / "experiment_logs"

# ── Experiment definitions ──────────────────────────────────────────────────

FORMATS = ["json", "toon", "tron"]

# Models for experiments (OpenRouter identifiers)
MODELS = {
    "qwen3-32b": {
        "benchpp": "qwen3-32b",
        "universe": "qwen/qwen3-32b",
        "mcpbench": "qwen-3-32b",
    },
    "gpt-4o-mini": {
        "benchpp": "gpt-4o-mini",
        "universe": "openai/gpt-4o-mini",
        "mcpbench": "gpt-4o-mini",
    },
    "deepseek-v3": {
        "benchpp": "deepseek-v3",
        "universe": "deepseek/deepseek-chat-v3-0324",
        "mcpbench": "deepseek-v3",
    },
}

LOCAL_MODEL = {
    "benchpp": "qwen3-32b",
    "universe": "qwen/qwen3-32b",
    "mcpbench": "qwen3-30b-openwebui",
}

# MCPToolBenchPP categories & input files
BENCHPP_CATEGORIES = [
    ("finance", "./data/finance/finance_0724_single_v3.json"),
    ("pay", "./data/pay/pay_0723_single.json"),
    ("search", "./data/search/search_0725_single_v2.json"),
    ("file_system", "./data/file_system/filesystem_0723_single.json"),
    ("map", "./data/map/map_0717_single_multi_lang_500.json"),
    ("browser", "./data/browser/browser_0724_single_v3.json"),
]

# MCP-Universe YAML config base dir
UNIVERSE_CONFIG_DIR = UNIVERSE_DIR / "mcpuniverse" / "benchmark" / "configs" / "mcpuniverse" / "test_full"
UNIVERSE_CATEGORIES = [
    "web_search", "financial_analysis", "location_navigation",
    "browser_automation", "repository_management", "3d_design",
]

# mcp-bench task files
MCPBENCH_TASK_FILES = [
    "tasks/mcpbench_tasks_single_runner_format.json",
    "tasks/mcpbench_tasks_multi_2server_runner_format.json",
    "tasks/mcpbench_tasks_multi_3server_runner_format.json",
]

JUDGE_MODEL_BENCHPP = "deepseek-v3.2"
JUDGE_MODEL_MCPBENCH = "deepseek/deepseek-v3.2"

# ── Time window for local LLM ──────────────────────────────────────────────

LOCAL_START_HOUR = 23  # 23:00
LOCAL_END_HOUR = 6     # 06:00


def in_local_window() -> bool:
    """Check if current time is within the local LLM window (23:00-06:00)."""
    h = datetime.now().hour
    return h >= LOCAL_START_HOUR or h < LOCAL_END_HOUR


def wait_for_local_window():
    """Block until the local LLM time window opens."""
    if in_local_window():
        return
    now = datetime.now()
    if now.hour < LOCAL_START_HOUR:
        target = now.replace(hour=LOCAL_START_HOUR, minute=0, second=0, microsecond=0)
    else:
        target = (now + timedelta(days=1)).replace(hour=LOCAL_START_HOUR, minute=0, second=0, microsecond=0)
    wait_secs = (target - now).total_seconds()
    print(f"  [local-llm] Outside time window. Waiting until {target.strftime('%H:%M')} ({wait_secs/3600:.1f}h)")
    time.sleep(wait_secs)


def check_local_window_or_stop() -> bool:
    """Return True if still in local window, False if we should stop."""
    return in_local_window()


# ── Benchmark runner helpers ────────────────────────────────────────────────

def run_cmd(cmd, cwd, log_file, env=None):
    """Run a subprocess, logging to file. Returns (returncode, duration)."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    start = time.time()
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as lf:
        lf.write(f"# CMD: {' '.join(str(c) for c in cmd)}\n")
        lf.write(f"# CWD: {cwd}\n")
        lf.write(f"# TIME: {datetime.now().isoformat()}\n\n")
        lf.flush()
        proc = subprocess.run(
            [str(c) for c in cmd],
            stdout=lf, stderr=subprocess.STDOUT,
            cwd=str(cwd), env=merged_env
        )
    duration = time.time() - start
    return proc.returncode, duration


# ── MCPToolBenchPP ──────────────────────────────────────────────────────────

def run_benchpp(fmt, model_key, tool_call_format=None, tag=""):
    """Run MCPToolBenchPP for all categories with given format and model."""
    model = MODELS[model_key]["benchpp"] if model_key in MODELS else model_key
    tc_fmt = tool_call_format or fmt
    results = []

    for category, input_file in BENCHPP_CATEGORIES:
        log_name = f"benchpp_{tag}_{model_key}_{fmt}_tc{tc_fmt}_{category}.log"
        log_path = str(LOG_DIR / log_name)

        cmd = [
            BENCHPP_PYTHON, "run.py",
            "--stage", "tool_call",
            "--input_file", input_file,
            "--category", category,
            "--model", model,
            "--tool_mode", "prompt",
            "--tool_format", fmt,
            "--tool_call_format", tc_fmt,
            "--llm_as_judge_model", JUDGE_MODEL_BENCHPP,
            "--pass_k", "1",
            "--evaluation_trial_per_task", "5",
        ]

        print(f"    [benchpp] {category} fmt={fmt} tc={tc_fmt} model={model}")
        rc, dur = run_cmd(cmd, BENCHPP_DIR, log_path)
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"    [benchpp] {category} -> {status} ({dur:.0f}s)")
        results.append({"benchmark": "benchpp", "category": category,
                         "format": fmt, "tc_format": tc_fmt, "model": model_key,
                         "rc": rc, "duration": dur, "log": log_path})
    return results


# ── MCP-Universe ────────────────────────────────────────────────────────────

def get_universe_config_path(category, fmt):
    """Get the YAML config path for a given category and format."""
    if fmt == "json":
        return UNIVERSE_CONFIG_DIR / f"{category}.yaml"
    return UNIVERSE_CONFIG_DIR / f"{category}_{fmt}.yaml"


def create_universe_config_variant(category, fmt, model_key, tool_call_format=None, tag=""):
    """Create a temporary YAML config with modified model and tool_call_format."""
    base_path = get_universe_config_path(category, fmt)
    if not base_path.exists():
        print(f"    [universe] WARNING: config not found: {base_path}")
        return None

    with open(base_path) as f:
        docs = list(yaml.safe_load_all(f))

    # Update model
    model_name = MODELS[model_key]["universe"] if model_key in MODELS else model_key
    for doc in docs:
        if doc.get("kind") == "llm":
            doc["spec"]["config"]["model_name"] = model_name
        if doc.get("kind") == "agent" and tool_call_format:
            doc["spec"]["config"]["tool_call_format"] = tool_call_format

    # Write to temp dir
    tmp_dir = LOG_DIR / "universe_configs" / tag
    os.makedirs(tmp_dir, exist_ok=True)
    tc_fmt = tool_call_format or fmt
    tmp_path = tmp_dir / f"{category}_{fmt}_tc{tc_fmt}_{model_key}.yaml"
    with open(tmp_path, "w") as f:
        yaml.dump_all(docs, f, default_flow_style=False)

    return tmp_path


def run_universe(fmt, model_key, tool_call_format=None, tag=""):
    """Run MCP-Universe for all categories with given format and model."""
    tc_fmt = tool_call_format or fmt
    results = []

    for category in UNIVERSE_CATEGORIES:
        config_path = create_universe_config_variant(
            category, fmt, model_key, tool_call_format=tool_call_format, tag=tag
        )
        if config_path is None:
            continue

        log_name = f"universe_{tag}_{model_key}_{fmt}_tc{tc_fmt}_{category}.log"
        log_path = str(LOG_DIR / log_name)

        cmd = [UNIVERSE_PYTHON, "run_full_test.py", str(config_path)]

        # MCP servers resolve "python3" via shutil.which(); prepend venv to PATH
        venv_bin = str(UNIVERSE_DIR / ".venv" / "bin")
        env = {"PATH": venv_bin + ":" + os.environ.get("PATH", "")}

        print(f"    [universe] {category} fmt={fmt} tc={tc_fmt} model={model_key}")
        rc, dur = run_cmd(cmd, UNIVERSE_DIR, log_path, env=env)
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"    [universe] {category} -> {status} ({dur:.0f}s)")
        results.append({"benchmark": "universe", "category": category,
                         "format": fmt, "tc_format": tc_fmt, "model": model_key,
                         "rc": rc, "duration": dur, "log": log_path})
    return results


# ── mcp-bench ───────────────────────────────────────────────────────────────

def run_mcpbench(fmt, model_key, tool_call_format=None, tag=""):
    """Run mcp-bench for all task files with given format and model."""
    model = MODELS[model_key]["mcpbench"] if model_key in MODELS else model_key
    tc_fmt = tool_call_format or fmt
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for tasks_file in MCPBENCH_TASK_FILES:
        task_label = Path(tasks_file).stem.replace("mcpbench_tasks_", "")
        log_name = f"mcpbench_{tag}_{model_key}_{fmt}_tc{tc_fmt}_{task_label}.log"
        log_path = str(LOG_DIR / log_name)
        output_file = f"results/gaia/bench_{fmt}_tc{tc_fmt}_{model_key}_{task_label}_{timestamp}.json"

        cmd = [
            MCPBENCH_PYTHON, "benchmark/runner.py",
            "--models", model,
            "--tasks-file", tasks_file,
            "--compression-format", fmt,
            "--tool-call-format", tc_fmt,
            "--output", output_file,
            "--skip-judge",
        ]

        print(f"    [mcpbench] {task_label} fmt={fmt} tc={tc_fmt} model={model}")
        rc, dur = run_cmd(cmd, MCPBENCH_DIR, log_path)
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"    [mcpbench] {task_label} -> {status} ({dur:.0f}s)")
        results.append({"benchmark": "mcpbench", "category": task_label,
                         "format": fmt, "tc_format": tc_fmt, "model": model_key,
                         "rc": rc, "duration": dur, "log": log_path})
    return results


# ── Experiment orchestration ────────────────────────────────────────────────

def run_experiment(exp_name, model_key="qwen3-32b", formats=None, local=False, benchmarks=None):
    """Run a single experiment across selected benchmarks."""
    if formats is None:
        formats = FORMATS
    if benchmarks is None:
        benchmarks = ["benchpp", "universe", "mcpbench"]

    all_results = []
    tag = exp_name

    if exp_name == "exp0":
        print(f"\n{'='*60}")
        print(f"Exp 0: Offline Token Analysis")
        print(f"{'='*60}")
        # Run the token analysis script (no API calls)
        token_script = ROOT / "token_analysis.py"
        if token_script.exists():
            log_path = str(LOG_DIR / "exp0_token_analysis.log")
            cmd = [sys.executable, str(token_script)]
            print("  Running token analysis...")
            rc, dur = run_cmd(cmd, ROOT, log_path)
            print(f"  Done ({dur:.0f}s) -> {'OK' if rc == 0 else 'FAIL'}")
        else:
            print("  token_analysis.py not found yet — skipping")
        return all_results

    if exp_name == "exp1":
        print(f"\n{'='*60}")
        print(f"Exp 1: Input Compression (schema+results compressed, tool calls JSON)")
        print(f"Model: {model_key}")
        print(f"{'='*60}")
        for fmt in formats:
            tc = "json"  # Tool calls always JSON in exp1
            if fmt == "json":
                continue  # json->json is baseline, no compression
            print(f"\n  Format: {fmt} (tool_call: {tc})")
            if "benchpp" in benchmarks:
                all_results.extend(run_benchpp(fmt, model_key, tool_call_format=tc, tag=tag))
            if "universe" in benchmarks:
                all_results.extend(run_universe(fmt, model_key, tool_call_format=tc, tag=tag))
            if "mcpbench" in benchmarks:
                all_results.extend(run_mcpbench(fmt, model_key, tool_call_format=tc, tag=tag))

        # Also run JSON baseline
        print(f"\n  Baseline: json (tool_call: json)")
        if "benchpp" in benchmarks:
            all_results.extend(run_benchpp("json", model_key, tool_call_format="json", tag=tag))
        if "universe" in benchmarks:
            all_results.extend(run_universe("json", model_key, tool_call_format="json", tag=tag))
        if "mcpbench" in benchmarks:
            all_results.extend(run_mcpbench("json", model_key, tool_call_format="json", tag=tag))

    elif exp_name == "exp2":
        print(f"\n{'='*60}")
        print(f"Exp 2: Full Compression (everything compressed)")
        print(f"Model: {model_key}")
        print(f"{'='*60}")
        for fmt in formats:
            print(f"\n  Format: {fmt} (tool_call: {fmt})")
            if "benchpp" in benchmarks:
                all_results.extend(run_benchpp(fmt, model_key, tool_call_format=fmt, tag=tag))
            if "universe" in benchmarks:
                all_results.extend(run_universe(fmt, model_key, tool_call_format=fmt, tag=tag))
            if "mcpbench" in benchmarks:
                all_results.extend(run_mcpbench(fmt, model_key, tool_call_format=fmt, tag=tag))

    elif exp_name == "exp3":
        print(f"\n{'='*60}")
        print(f"Exp 3: Model Comparison (full compression)")
        print(f"Models: {', '.join(MODELS.keys())}")
        print(f"{'='*60}")
        for mk in MODELS:
            for fmt in formats:
                print(f"\n  Model: {mk}, Format: {fmt}")
                if "benchpp" in benchmarks:
                    all_results.extend(run_benchpp(fmt, mk, tool_call_format=fmt, tag=tag))
                if "universe" in benchmarks:
                    all_results.extend(run_universe(fmt, mk, tool_call_format=fmt, tag=tag))
                if "mcpbench" in benchmarks:
                    all_results.extend(run_mcpbench(fmt, mk, tool_call_format=fmt, tag=tag))

    return all_results


def run_local_experiment(model_key="qwen3-32b", formats=None):
    """Run experiments with local LLM (time-windowed 23:00-06:00)."""
    if formats is None:
        formats = FORMATS

    print(f"\n{'='*60}")
    print(f"Local LLM Experiment (time window: {LOCAL_START_HOUR}:00-{LOCAL_END_HOUR:02d}:00)")
    print(f"{'='*60}")

    wait_for_local_window()
    tag = "local"
    all_results = []

    for fmt in formats:
        if not check_local_window_or_stop():
            print(f"\n  TIME WINDOW CLOSED — stopping local experiments")
            break

        print(f"\n  Format: {fmt}")
        # Use local model identifiers
        log_name = f"local_{fmt}_benchpp.log"
        # MCPToolBenchPP with local LLM
        for category, input_file in BENCHPP_CATEGORIES:
            if not check_local_window_or_stop():
                break
            log_path = str(LOG_DIR / f"local_benchpp_{fmt}_{category}.log")
            cmd = [
                BENCHPP_PYTHON, "run.py",
                "--stage", "tool_call",
                "--input_file", input_file,
                "--category", category,
                "--model", LOCAL_MODEL["benchpp"],
                "--tool_mode", "prompt",
                "--tool_format", fmt,
                "--tool_call_format", fmt,
                "--llm_as_judge_model", JUDGE_MODEL_BENCHPP,
                "--pass_k", "1",
                "--evaluation_trial_per_task", "5",
            ]
            print(f"    [local/benchpp] {category} fmt={fmt}")
            rc, dur = run_cmd(cmd, BENCHPP_DIR, log_path)
            all_results.append({"benchmark": "benchpp", "category": category,
                                 "format": fmt, "model": "local", "rc": rc,
                                 "duration": dur, "log": log_path})

        # mcp-bench with local LLM
        for tasks_file in MCPBENCH_TASK_FILES:
            if not check_local_window_or_stop():
                break
            task_label = Path(tasks_file).stem.replace("mcpbench_tasks_", "")
            log_path = str(LOG_DIR / f"local_mcpbench_{fmt}_{task_label}.log")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cmd = [
                MCPBENCH_PYTHON, "benchmark/runner.py",
                "--models", LOCAL_MODEL["mcpbench"],
                "--tasks-file", tasks_file,
                "--compression-format", fmt,
                "--tool-call-format", fmt,
                "--output", f"results/gaia/local_{fmt}_{task_label}_{timestamp}.json",
                "--skip-judge",
            ]
            print(f"    [local/mcpbench] {task_label} fmt={fmt}")
            rc, dur = run_cmd(cmd, MCPBENCH_DIR, log_path)
            all_results.append({"benchmark": "mcpbench", "category": task_label,
                                 "format": fmt, "model": "local", "rc": rc,
                                 "duration": dur, "log": log_path})

        # MCP-Universe with local LLM
        for category in UNIVERSE_CATEGORIES:
            if not check_local_window_or_stop():
                break
            config_path = create_universe_config_variant(
                category, fmt, model_key, tool_call_format=fmt, tag="local"
            )
            if config_path is None:
                continue
            log_path = str(LOG_DIR / f"local_universe_{fmt}_{category}.log")
            cmd = [UNIVERSE_PYTHON, "run_full_test.py", str(config_path)]
            print(f"    [local/universe] {category} fmt={fmt}")
            rc, dur = run_cmd(cmd, UNIVERSE_DIR, log_path)
            all_results.append({"benchmark": "universe", "category": category,
                                 "format": fmt, "model": "local", "rc": rc,
                                 "duration": dur, "log": log_path})

    return all_results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified experiment runner for MCP format compression paper"
    )
    parser.add_argument(
        "--exp", nargs="+", required=True,
        choices=["exp0", "exp1", "exp2", "exp3", "all"],
        help="Experiments to run"
    )
    parser.add_argument(
        "--model", default="qwen3-32b",
        help="Default model for exp1/exp2 (default: qwen3-32b)"
    )
    parser.add_argument(
        "--formats", nargs="+", default=FORMATS,
        choices=["json", "toon", "tron"],
        help="Formats to test (default: json toon tron)"
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Include local LLM run (time-windowed 23:00-06:00)"
    )
    parser.add_argument(
        "--benchmark", nargs="+", default=["benchpp", "universe", "mcpbench"],
        choices=["benchpp", "universe", "mcpbench"],
        help="Which benchmarks to run (default: all three)"
    )

    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    experiments = args.exp
    if "all" in experiments:
        experiments = ["exp0", "exp1", "exp2", "exp3"]

    print(f"{'='*60}")
    print(f"MCP Format Compression — Experiment Runner")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiments: {', '.join(experiments)}")
    print(f"Model: {args.model}")
    print(f"Formats: {', '.join(args.formats)}")
    print(f"Benchmarks: {', '.join(args.benchmark)}")
    print(f"Local LLM: {'yes' if args.local else 'no'}")
    print(f"{'='*60}")

    all_results = []
    for exp in experiments:
        results = run_experiment(exp, model_key=args.model, formats=args.formats,
                                benchmarks=args.benchmark)
        all_results.extend(results)

    if args.local:
        local_results = run_local_experiment(model_key=args.model, formats=args.formats)
        all_results.extend(local_results)

    # Save results summary
    summary_path = LOG_DIR / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"{'Benchmark':<12} {'Category':<24} {'Format':<6} {'TC':<6} {'Model':<14} {'Status':<8} {'Time':>8}")
    print(f"{'-'*12} {'-'*24} {'-'*6} {'-'*6} {'-'*14} {'-'*8} {'-'*8}")

    for r in all_results:
        status = "OK" if r["rc"] == 0 else f"FAIL"
        tc = r.get("tc_format", r.get("format", "?"))
        print(f"{r['benchmark']:<12} {r['category']:<24} {r['format']:<6} {tc:<6} {r['model']:<14} {status:<8} {r['duration']:>7.0f}s")

    total_dur = sum(r["duration"] for r in all_results)
    n_ok = sum(1 for r in all_results if r["rc"] == 0)
    n_fail = sum(1 for r in all_results if r["rc"] != 0)
    print(f"\nTotal: {len(all_results)} runs, {n_ok} OK, {n_fail} FAIL, {total_dur:.0f}s ({total_dur/3600:.1f}h)")
    print(f"Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
