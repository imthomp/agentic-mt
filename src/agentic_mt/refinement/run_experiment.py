"""TEaR self-refinement (condition A) vs. COMET-KIWI-guided refinement
(condition B), 4 iterations, en-de.

Usage:
    # Generation (SLURM, one job runs both conditions on one GPU):
    uv run python -m agentic_mt.refinement.run_experiment --config configs/refinement_tear.yaml --generate

    # Analysis (after outputs.csv exists):
    uv run python -m agentic_mt.refinement.run_experiment --config configs/refinement_tear.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from agentic_mt.pipelines.data import sample_test_and_tm_corpus
from agentic_mt.pipelines.scoring import score_all
from agentic_mt.refinement import condition_a, condition_b
from agentic_mt.refinement.analysis import iteration_means, peak_analysis, plot_quality_vs_iteration
from agentic_mt.refinement.iterate import run_iterative

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

SRC_LANG, TGT_LANG = "English", "German"


def generate(config: dict) -> Path:
    n_test = config["sample_size"]
    seed = config["random_seed"]
    model_name = config["model_name"]
    n_iterations = config["n_iterations"]
    qe_threshold = config.get("qe_threshold", condition_b.DEFAULT_THRESHOLD)

    out_dir = Path(config["output_dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same seed as Phase 3's en-de sample -> the same 100 test sentences.
    test_df, _ = sample_test_and_tm_corpus("en-de", n_test, seed)

    import functools
    step_b = functools.partial(condition_b.step, threshold=qe_threshold)

    logger.info("Running condition A (TEaR self-refinement)")
    df_a, log_a = run_iterative(test_df, condition_a.step, "condition_a_tear", SRC_LANG, TGT_LANG, model_name, n_iterations)

    logger.info("Running condition B (COMET-KIWI-guided refinement)")
    df_b, log_b = run_iterative(test_df, step_b, "condition_b_qe", SRC_LANG, TGT_LANG, model_name, n_iterations)

    with (out_dir / "prompts.jsonl").open("w") as f:
        for entry in log_a + log_b:
            f.write(json.dumps(entry) + "\n")

    from agentic_mt.pipelines.llm import unload as unload_llm
    unload_llm()

    df = pd.concat([df_a, df_b], ignore_index=True)
    logger.info(f"Scoring {len(df)} (sentence, condition, iteration) outputs")
    scores = score_all(df["source"].tolist(), df["translation"].tolist(), df["reference"].tolist(), gpus=1)
    for metric, values in scores.items():
        df[metric] = values

    out_path = out_dir / "outputs.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path}")
    return out_path


def analyze(config: dict) -> Path:
    metrics = config.get("metrics", ["cometkiwi", "comet", "chrf"])
    out_dir = Path(config["output_dir"]).expanduser()

    df = pd.read_csv(out_dir / "outputs.csv")
    means = iteration_means(df, metrics)
    means.to_csv(out_dir / "iteration_means.csv", index=False)

    peak_frames = [peak_analysis(means, metric) for metric in metrics]
    peak_df = pd.concat(peak_frames, ignore_index=True)
    peak_df.to_csv(out_dir / "peak_analysis.csv", index=False)

    plot_quality_vs_iteration(means, metrics, out_dir / "quality_vs_iteration.png")

    report_path = out_dir / "report.md"
    lines = ["# TEaR vs. QE-guided refinement\n", "## Mean score per condition per iteration\n",
             means.to_markdown(index=False), "\n## Iteration-of-peak and gain (iterations 1..N only)\n",
             peak_df.to_markdown(index=False)]
    report_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote iteration means to {out_dir / 'iteration_means.csv'}")
    print(f"Wrote peak analysis to {out_dir / 'peak_analysis.csv'}")
    print(f"Wrote plot to {out_dir / 'quality_vs_iteration.png'}")
    print(f"Wrote report to {report_path}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--generate", action="store_true", help="Run generation+scoring (SLURM GPU mode).")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.generate:
        generate(cfg)
    else:
        analyze(cfg)
