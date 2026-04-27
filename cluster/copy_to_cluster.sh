#!/bin/bash
# Copy only the required files to the cluster
# Usage: bash cluster/copy_to_cluster.sh user@cluster.example.com
set -e

REMOTE="${1:-lkutschka@vsc5.vsc.ac.at}"
REMOTE_DIR="~/phd"

echo "Copying project files to $REMOTE:$REMOTE_DIR"

# Create remote directory structure
ssh "$REMOTE" "mkdir -p $REMOTE_DIR/{cluster,logs}"

# Cluster scripts
rsync -av --progress \
    cluster/ "$REMOTE:$REMOTE_DIR/cluster/"

# StableToolBench (benchmark code + cache, no results)
rsync -av --progress \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='results/' \
    --exclude='eval_results/' \
    --exclude='eval_converted/' \
    --exclude='*.log' \
    --exclude='.git' \
    StableToolBench/ "$REMOTE:$REMOTE_DIR/StableToolBench/"

# Shared libraries
for lib in shared_format toon-python tron-python; do
    rsync -av --progress \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='*.egg-info' \
        "$lib/" "$REMOTE:$REMOTE_DIR/$lib/"
done

echo ""
echo "Done! Now SSH to the cluster and run:"
echo "  cd $REMOTE_DIR"
echo "  bash cluster/setup.sh"
echo "  bash cluster/run_all.sh"
