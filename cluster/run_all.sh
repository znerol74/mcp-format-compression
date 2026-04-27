#!/bin/bash
# Submit SLURM jobs for all models in models.txt
# Usage: bash cluster/run_all.sh [benchmark]
#   benchmark: stabletoolbench (default) or bfcl
set -e

cd "$(dirname "$0")/.."

BENCHMARK="${1:-stabletoolbench}"

echo "Submitting jobs for benchmark: $BENCHMARK"
echo "==========================================="

while IFS=: read -r HF_MODEL TAG NGPU QUANT PARTITION; do
    [[ "$HF_MODEL" =~ ^# ]] && continue
    [ -z "$HF_MODEL" ] && continue

    echo "  $TAG ($HF_MODEL)"
    echo "    Partition: $PARTITION, GPUs: $NGPU, Quant: $QUANT"

    JOB_ID=$(sbatch --parsable \
        --partition="$PARTITION" \
        --qos="$PARTITION" \
        --gres="gpu:$NGPU" \
        --job-name="fmt-${TAG}" \
        --export="MODEL_HF=$HF_MODEL,MODEL_TAG=$TAG,QUANT=$QUANT,BENCHMARK=$BENCHMARK" \
        cluster/run_model.slurm)

    echo "    Submitted: job $JOB_ID"
    echo ""
done < cluster/models.txt

echo "==========================================="
echo "All jobs submitted!"
echo "Monitor with: squeue -u $USER"
echo "Logs in:      logs/"
