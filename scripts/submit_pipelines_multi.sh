#!/bin/bash
# Generalized version of submit_pipelines.sh: reads the `pairs` list from
# the given config and submits one parallel job per pair (rather than
# hardcoding en-de/en-ha), then a dependent merge+analysis job. Used for
# Phase 5's multi-pair BLOOMZ/InkubaLM configs.
#
# Usage: bash scripts/submit_pipelines_multi.sh <config>

set -e
cd "$(dirname "$0")/.."

CONFIG=${1:?Usage: submit_pipelines_multi.sh <config>}
PARTITION=dwmatrix
QOS=matrix
VENV=.venv/bin/python
SCRIPT=src/agentic_mt/pipelines/run_experiment.py
LOG=logs

mkdir -p "$LOG"

PAIRS=$($VENV -c "import yaml; print(' '.join(yaml.safe_load(open('$CONFIG'))['pairs']))")
echo "Pairs from $CONFIG: $PAIRS"

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
JOB_IDS=()
for pair in $PAIRS; do
    jid=$(submit "$pair" 06:00:00 128G "--pair $pair")
    echo "  $pair: $jid (GPU, 6h, 128G)"
    JOB_IDS+=("$jid")
done

DEP=$(IFS=:; echo "${JOB_IDS[*]}")
JOB_ANALYSIS=$(sbatch \
    --dependency=afterok:${DEP} \
    --partition=$PARTITION --qos=$QOS \
    --time=00:30:00 --mem=32G \
    --ntasks=1 --nodes=1 \
    --job-name="agentic_mt_pipe_analysis" \
    --output="${LOG}/pipe_analysis_%j.out" \
    --wrap="cd \$SLURM_SUBMIT_DIR && \
            export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && \
            $VENV -u $SCRIPT --config $CONFIG" \
    | awk '{print $NF}')
echo "  analysis: $JOB_ANALYSIS (afterok all pairs, CPU only, 30m)"

echo ""
echo "Monitor: squeue -u $USER"
