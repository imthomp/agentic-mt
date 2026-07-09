"""Four translation conditions (L1_baseline, L1_cot, L1_tool, L1_cot_tool)
on the same source sentences / same model, for a high-resource pair (en-de)
and a low-resource pair (en-ha).

Usage:
    # Per-pair generation + scoring (SLURM array-task mode, one GPU per pair):
    uv run python -m agentic_mt.pipelines.run_experiment --config configs/pipelines_l1_conditions.yaml --pair en-de

    # Merge + analysis (after both pairs' outputs.csv exist):
    uv run python -m agentic_mt.pipelines.run_experiment --config configs/pipelines_l1_conditions.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from agentic_mt.pipelines.analysis import condition_means_table, pairwise_significance_table, resource_gap_comparison
from agentic_mt.pipelines.conditions import CONDITIONS
from agentic_mt.pipelines.data import sample_test_and_tm_corpus
from agentic_mt.pipelines.retrieval import TMIndex
from agentic_mt.pipelines.scoring import score_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_LANG_NAME = {"en-de": "German", "en-ha": "Hausa", "en-xh": "Xhosa", "en-zu": "Zulu", "en-sw": "Swahili"}
SOURCE_LANG_NAME = "English"

# Conditions that need a retrieved TM match.
TOOL_CONDITIONS = {"l1_tool", "l1_cot_tool"}


def run_pair(config: dict, pair: str) -> Path:
    n_test = config["sample_size"]
    seed = config["random_seed"]
    model_name = config["model_name"]
    conditions = config.get("conditions", list(CONDITIONS))
    gpus = config.get("gpus", 1)
    retrieval_device = config.get("retrieval_device")

    out_dir = Path(config["output_dir"]).expanduser() / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Sampling {n_test} test sentences + TM corpus for {pair}")
    test_df, tm_corpus_df = sample_test_and_tm_corpus(pair, n_test, seed)
    logger.info(f"{pair}: {len(test_df)} test sentences, {len(tm_corpus_df)} TM corpus sentences")

    tm_index = TMIndex(tm_corpus_df, device=retrieval_device)
    tgt_lang = TARGET_LANG_NAME[pair]

    rows = []
    prompt_log_path = out_dir / f"prompts_{pair}.jsonl"
    with prompt_log_path.open("w") as log_f:
        for sentence_idx, row in test_df.iterrows():
            source_text = row["source"]
            reference = row["reference"]
            tm_match = tm_index.retrieve(source_text, k=1)[0]

            for condition_name in conditions:
                condition_fn = CONDITIONS[condition_name]
                match_arg = tm_match if condition_name in TOOL_CONDITIONS else None
                result = condition_fn(source_text, SOURCE_LANG_NAME, tgt_lang, model_name, match_arg)

                log_f.write(json.dumps({
                    "pair": pair, "sentence_idx": int(sentence_idx), "condition": condition_name,
                    "source": source_text, "reference": reference, **result,
                }) + "\n")

                rows.append({
                    "pair": pair, "sentence_idx": int(sentence_idx), "condition": condition_name,
                    "source": source_text, "reference": reference,
                    "final_translation": result["final_translation"],
                    "prompt_char_len": sum(len(t["prompt"]) for t in result["turns"]),
                })
            log_f.flush()
            logger.info(f"{pair}: sentence {sentence_idx} done ({len(conditions)} conditions)")

    from agentic_mt.pipelines.llm import unload as unload_llm
    unload_llm()

    df = pd.DataFrame(rows)
    logger.info(f"Scoring {len(df)} (sentence, condition) outputs for {pair}")
    scores = score_all(df["source"].tolist(), df["final_translation"].tolist(), df["reference"].tolist(), gpus=gpus)
    for metric, values in scores.items():
        df[metric] = values

    out_path = out_dir / f"outputs_{pair}.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path}")
    return out_path


def run_analysis(config: dict) -> Path:
    pairs = config["pairs"]
    high_resource_pair = config["high_resource_pair"]
    low_resource_pairs = config.get("low_resource_pairs") or [config["low_resource_pair"]]
    metrics = config.get("metrics", ["cometkiwi", "comet", "chrf"])
    n_bootstrap = config.get("n_bootstrap", 1000)
    seed = config["random_seed"]

    runs_dir = Path(config["output_dir"]).expanduser() / "runs"
    out_dir = Path(config["output_dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [pd.read_csv(runs_dir / f"outputs_{pair}.csv") for pair in pairs]
    df = pd.concat(frames, ignore_index=True)

    means = condition_means_table(df, metrics)
    means.to_csv(out_dir / "condition_means.csv", index=False)

    sig_frames = [pairwise_significance_table(df, metric, n_bootstrap, seed) for metric in metrics]
    sig_df = pd.concat(sig_frames, ignore_index=True)
    sig_df.to_csv(out_dir / "pairwise_significance.csv", index=False)

    gap_frames = []
    for low_resource_pair in low_resource_pairs:
        for metric in metrics:
            gap_frames.append(
                resource_gap_comparison(df, metric, high_resource_pair, low_resource_pair, n_resamples=n_bootstrap, seed=seed)
            )
    gap_df = pd.concat(gap_frames, ignore_index=True)
    gap_df.to_csv(out_dir / "resource_gap_comparison.csv", index=False)

    report_path = out_dir / "report.md"
    _write_report(report_path, means, sig_df, gap_df, high_resource_pair, low_resource_pairs)

    print(f"Wrote condition means to {out_dir / 'condition_means.csv'}")
    print(f"Wrote pairwise significance to {out_dir / 'pairwise_significance.csv'}")
    print(f"Wrote resource gap comparison to {out_dir / 'resource_gap_comparison.csv'}")
    print(f"Wrote report to {report_path}")
    return out_dir


def _write_report(path: Path, means: pd.DataFrame, sig_df: pd.DataFrame, gap_df: pd.DataFrame,
                   high_resource_pair: str, low_resource_pairs: list[str]) -> None:
    lines = ["# L1 translation-condition experiment\n"]
    lines.append(f"High-resource pair: {high_resource_pair}. Low-resource pair(s): {', '.join(low_resource_pairs)}.\n")
    lines.append("## Mean score per condition per pair\n")
    lines.append(means.to_markdown(index=False))
    lines.append("\n## Paired bootstrap significance between conditions\n")
    lines.append(sig_df.to_markdown(index=False))
    lines.append("\n## Core hypothesis test: does the condition-4 gap differ by resource level?\n")
    lines.append(gap_df.to_markdown(index=False))
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pair", default=None,
                        help="If set, only generate+score this language pair (SLURM array-task mode); "
                             "otherwise run the merge+analysis step over all configured pairs.")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.pair:
        run_pair(cfg, args.pair)
    else:
        run_analysis(cfg)
