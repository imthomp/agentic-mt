#!/bin/bash
# Submit the triggering-feature pipeline in PARALLEL on dwmatrix (8x A100).
# One job per language does the full feature extraction (including LLM-based
# idiomaticity scoring, the GPU-bound step) for that language's full row set.
# A final merge/modeling job runs after all five complete, joining the
# per-language feature CSVs and fitting the logistic regression models.
#
# Mirrors multilingual-mqm-benchmark/scripts/slurm/submit_chain.sh conventions,
# including its hard-won lesson: even though a language's data is a few MB,
# request >=128G — compute-node cgroup accounting adds ~60-100GB of overhead
# for the CUDA context + Llama-3.1-8B weights that doesn't show up when
# profiling on the login node.
#
# Usage: bash scripts/submit_triggering.sh [config]

set -e
cd "$(dirname "$0")/.."

CONFIG=${1:-configs/triggering_low_resource.yaml}
PARTITION=dwmatrix
QOS=matrix
VENV=.venv/bin/python
SCRIPT=src/agentic_mt/triggering/run_experiment.py
LOG=logs

mkdir -p "$LOG"

submit() {
    local name=$1 time=$2 mem=$3 dep_flag=$4 extra_args=$5
    sbatch $dep_flag \
        --partition=$PARTITION --qos=$QOS \
        --time=$time --mem=$mem --gres=gpu:1 \
        --ntasks=1 --nodes=1 \
        --job-name="agentic_mt_trig_${name}" \
        --output="${LOG}/trig_${name}_%j.out" \
        --wrap="cd \$SLURM_SUBMIT_DIR && \
                export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && \
                $VENV -u $SCRIPT --config $CONFIG $extra_args" \
        | awk '{print $NF}'
}

echo "Submitting per-language feature extraction on $PARTITION (parallel, 1 GPU each)..."

# ha is much larger (23,983 rows across en-ha + ha-en) than the others; give
# it more wall time. The rest (km/ps ~4.6-4.7k, xh/zu ~2.5-3k) get 4h.
JOB_HA=$(submit ha 08:00:00 128G "" "--lang ha")
echo "  ha: $JOB_HA (GPU, 8h, 128G)"
JOB_KM=$(submit km 04:00:00 128G "" "--lang km")
echo "  km: $JOB_KM (GPU, 4h, 128G)"
JOB_PS=$(submit ps 04:00:00 128G "" "--lang ps")
echo "  ps: $JOB_PS (GPU, 4h, 128G)"
JOB_XH=$(submit xh 04:00:00 128G "" "--lang xh")
echo "  xh: $JOB_XH (GPU, 4h, 128G)"
JOB_ZU=$(submit zu 04:00:00 128G "" "--lang zu")
echo "  zu: $JOB_ZU (GPU, 4h, 128G)"

ALL_JOBS="${JOB_HA}:${JOB_KM}:${JOB_PS}:${JOB_XH}:${JOB_ZU}"
JOB_MERGE=$(sbatch \
    --dependency=afterok:${ALL_JOBS} \
    --partition=$PARTITION --qos=$QOS \
    --time=01:00:00 --mem=32G \
    --ntasks=1 --nodes=1 \
    --job-name="agentic_mt_trig_merge" \
    --output="${LOG}/trig_merge_%j.out" \
    --wrap="cd \$SLURM_SUBMIT_DIR && \
            export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && \
            $VENV -u $SCRIPT --config $CONFIG" \
    | awk '{print $NF}')
echo "  merge+model: $JOB_MERGE (afterok all 5, CPU only, 1h)"

echo ""
echo "All 5 languages extracting in parallel. Merge+modeling starts when all complete."
echo "Monitor: squeue -u $USER"
