#!/bin/bash
# Submit a chain of N 72h jobs, each depending on the previous (afterany).
# Each job resumes from its predecessor's state via deterministic result paths / log files.
#
# Usage:
#   bash cluster/submit_chain.sh <N> <slurm_script> <gres> <job_name> <EXPORT_STRING>
#
# Example (BenchPP for qwen3-32b-fp16, chain of 3, 3 GPUs each):
#   bash cluster/submit_chain.sh 3 cluster/run_benchpp.slurm gpu:3 benchpp-qwen3-32b-fp16 \
#       MODEL_HF=Qwen/Qwen3-32B,MODEL_TAG=qwen3-32b-fp16,QUANT=none,MODEL_TP=2

set -e

N="${1:?N required}"
SCRIPT="${2:?slurm script required}"
GRES="${3:?gres required (e.g. gpu:2)}"
JOBNAME="${4:?job name required}"
EXPORT="${5:?export string required}"

PARTITION="${PARTITION:-zen3_0512_a100x2}"

prev=""
ids=()
for i in $(seq 1 "$N"); do
    dep=""
    [ -n "$prev" ] && dep="--dependency=afterany:$prev"
    id=$(sbatch --parsable \
        --time=72:00:00 \
        --partition="$PARTITION" --qos="$PARTITION" --gres="$GRES" \
        --job-name="${JOBNAME}" \
        $dep \
        --export="$EXPORT" \
        "$SCRIPT")
    echo "  link $i/$N: $id $([ -n "$prev" ] && echo "(after $prev)")"
    ids+=("$id")
    prev="$id"
done

echo "Chain ids for $JOBNAME: ${ids[*]}"
