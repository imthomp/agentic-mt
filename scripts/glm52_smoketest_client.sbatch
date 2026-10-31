#!/bin/bash
#SBATCH --job-name=agentic_mt_glm52_smoketest
#SBATCH --partition=dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/glm52_smoketest_%j.out

# CPU-only client: polls for the GLM-5.2 vLLM server (launched separately
# via /nobackup/autodelete/usr/it238/glm52/serve_glm52.sbatch) and queries
# it over HTTP -- no GPU or vllm needed on this side, just this project's
# own venv (for the data loaders) and stdlib urllib.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -u scripts/glm52_exposure_floor_smoketest.py
