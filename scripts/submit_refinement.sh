#!/bin/bash
# Submit the TEaR-vs-QE-guided refinement experiment on dwmatrix. One GPU
# job runs both conditions (they share one loaded model), then a dependent
# CPU job runs the analysis. Mirrors submit_triggering.sh/submit_pipelines.sh
# conventions.
#
# Usage: bash scripts/submit_refinement.sh [config]

set -e
cd "$(dirname "$0")/.."

CONFIG=${1:-configs/refinement_tear.yaml}
PARTITION=dwmatrix
QOS=matrix
VENV=.venv/bin/python
SCRIPT=src/agentic_mt/refinement/run_experiment.py
LOG=logs

mkdir -p "$LOG"

JOB_GEN=$(sbatch \
    --partition=$PARTITION --qos=$QOS \
    --time=06:00:00 --mem=128G --gres=gpu:1 \
    --ntasks=1 --nodes=1 \
    --job-name="agentic_mt_refine_gen" \
    --output="${LOG}/refine_gen_%j.out" \
    --wrap="cd \$SLURM_SUBMIT_DIR && \
            export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && \
            $VENV -u $SCRIPT --config $CONFIG --generate" \
    | awk '{print $NF}')
echo "  generate: $JOB_GEN (GPU, 6h, 128G)"

JOB_ANALYSIS=$(sbatch \
    --dependency=afterok:${JOB_GEN} \
    --partition=$PARTITION --qos=$QOS \
    --time=00:15:00 --mem=16G \
    --ntasks=1 --nodes=1 \
    --job-name="agentic_mt_refine_analysis" \
    --output="${LOG}/refine_analysis_%j.out" \
    --wrap="cd \$SLURM_SUBMIT_DIR && \
            $VENV -u $SCRIPT --config $CONFIG" \
    | awk '{print $NF}')
echo "  analysis: $JOB_ANALYSIS (afterok generate, CPU only, 15m)"

echo ""
echo "Monitor: squeue -u $USER"
