#!/bin/bash
# Submit the L1 translation-condition pipeline in PARALLEL on dwmatrix.
# One job per language pair (en-de, en-ha) generates+scores all 4 conditions
# for that pair's 100 sampled sentences; a final analysis job runs after both
# complete. Mirrors scripts/submit_triggering.sh conventions.
#
# Usage: bash scripts/submit_pipelines.sh [config]

set -e
cd "$(dirname "$0")/.."

CONFIG=${1:-configs/pipelines_l1_conditions.yaml}
PARTITION=dwmatrix
QOS=matrix
VENV=.venv/bin/python
SCRIPT=src/agentic_mt/pipelines/run_experiment.py
LOG=logs

mkdir -p "$LOG"

submit() {
    local name=$1 time=$2 mem=$3 extra_args=$4
    sbatch \
        --partition=$PARTITION --qos=$QOS \
        --time=$time --mem=$mem --gres=gpu:1 \
        --ntasks=1 --nodes=1 \
        --job-name="agentic_mt_pipe_${name}" \
        --output="${LOG}/pipe_${name}_%j.out" \
        --wrap="cd \$SLURM_SUBMIT_DIR && \
                export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && \
                $VENV -u $SCRIPT --config $CONFIG $extra_args" \
        | awk '{print $NF}'
}

echo "Submitting per-pair generation+scoring on $PARTITION (parallel, 1 GPU each)..."
JOB_ENDE=$(submit ende 04:00:00 128G "--pair en-de")
echo "  en-de: $JOB_ENDE (GPU, 4h, 128G)"
JOB_ENHA=$(submit enha 04:00:00 128G "--pair en-ha")
echo "  en-ha: $JOB_ENHA (GPU, 4h, 128G)"

ALL_JOBS="${JOB_ENDE}:${JOB_ENHA}"
JOB_ANALYSIS=$(sbatch \
    --dependency=afterok:${ALL_JOBS} \
    --partition=$PARTITION --qos=$QOS \
    --time=00:30:00 --mem=16G \
    --ntasks=1 --nodes=1 \
    --job-name="agentic_mt_pipe_analysis" \
    --output="${LOG}/pipe_analysis_%j.out" \
    --wrap="cd \$SLURM_SUBMIT_DIR && \
            export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && \
            $VENV -u $SCRIPT --config $CONFIG" \
    | awk '{print $NF}')
echo "  analysis: $JOB_ANALYSIS (afterok both, CPU only, 30m)"

echo ""
echo "Both pairs generating in parallel. Analysis starts when both complete."
echo "Monitor: squeue -u $USER"
