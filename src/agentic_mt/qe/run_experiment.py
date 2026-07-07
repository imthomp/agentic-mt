"""QE-vs-human-judgment experiment for low-resource language pairs.

Reuses real WMT Direct Assessment human scores and pre-computed COMET-KIWI /
COMET / chrF scores from the multilingual-mqm-benchmark project (see
data.py for how the two are safely joined). For each low-resource pair,
computes Pearson/Spearman correlation between each metric and the human
score, with 1000-resample bootstrap confidence intervals, and flags any
pair where the reference-free QE metric (COMET-KIWI) falls below the 0.3
correlation threshold as a limitation, not something to route around.

Usage:
    uv run python -m agentic_mt.qe.run_experiment --config configs/qe_wmt_da_low_resource.yaml
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from agentic_mt.qe.data import build_merged_dataset
from agentic_mt.qe.stats import compute_correlations

QE_FLAG_THRESHOLD = 0.3


def run(config: dict) -> Path:
    mqmbench_root = Path(config["mqmbench_root"]).expanduser()
    target_langs = config["target_langs"]
    qe_metric = config["qe_metric"]
    reference_metrics = config["reference_metrics"]
    n_bootstrap = config["n_bootstrap"]
    seed = config["random_seed"]
    resource_notes = config.get("resource_tier_notes", {})

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = Path(config["output_dir"]).expanduser() / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_merged_dataset(mqmbench_root, target_langs, metrics=[qe_metric, *reference_metrics])

    rows = []
    flags = []
    for lang in target_langs:
        sub = df[df["lang"] == lang]
        human = sub["quality_score"].to_numpy()

        for metric in [qe_metric, *reference_metrics]:
            corr = compute_correlations(sub[metric].to_numpy(), human, n_resamples=n_bootstrap, seed=seed)
            rows.append({"lang": lang, "metric": metric, "reference_free": metric == qe_metric, **corr})

            if metric == qe_metric and abs(corr["spearman_r"]) < QE_FLAG_THRESHOLD:
                flags.append(
                    f"- **{lang}** ({resource_notes.get(lang, '')}): {qe_metric} vs. human Spearman "
                    f"r = {corr['spearman_r']:.3f} (95% CI [{corr['spearman_ci_lo']:.3f}, "
                    f"{corr['spearman_ci_hi']:.3f}], n={corr['n']}) — below {QE_FLAG_THRESHOLD}. "
                    "QE-in-the-loop refinement is not currently viable for this pair without "
                    "QE adaptation; report this as a limitation rather than routing around it."
                )

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    _plot_scatter(df, target_langs, qe_metric, out_dir / "scatter.png")

    report_path = out_dir / "report.md"
    _write_report(report_path, summary, flags, qe_metric, resource_notes)

    print(f"Wrote summary table to {summary_path}")
    print(f"Wrote scatter plot to {out_dir / 'scatter.png'}")
    print(f"Wrote report to {report_path}")
    if flags:
        print("\n=== LIMITATION FLAGGED ===")
        for f in flags:
            print(f)

    return out_dir


def _plot_scatter(df: pd.DataFrame, target_langs: list[str], qe_metric: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(target_langs), figsize=(4 * len(target_langs), 4), sharey=True)
    if len(target_langs) == 1:
        axes = [axes]
    for ax, lang in zip(axes, target_langs):
        sub = df[df["lang"] == lang]
        ax.scatter(sub[qe_metric], sub["quality_score"], s=6, alpha=0.25)
        ax.set_title(lang)
        ax.set_xlabel(qe_metric)
    axes[0].set_ylabel("human quality score (normalized DA)")
    fig.suptitle(f"{qe_metric} (reference-free QE) vs. human DA score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_report(path: Path, summary: pd.DataFrame, flags: list[str], qe_metric: str, resource_notes: dict) -> None:
    lines = ["# QE vs. human-judgment correlation — low-resource pairs\n"]
    lines.append(summary.to_markdown(index=False))
    lines.append("\n## Limitations\n")
    if flags:
        lines.append(
            f"The following pairs show {qe_metric}-vs-human Spearman correlation below "
            f"{QE_FLAG_THRESHOLD}, a strong signal that QE-in-the-loop refinement is not "
            "currently viable for these pairs without QE adaptation:\n"
        )
        lines.extend(flags)
    else:
        lines.append(f"No pair fell below the {QE_FLAG_THRESHOLD} correlation threshold for {qe_metric}.")
    if resource_notes:
        lines.append("\n## Resource-tier notes\n")
        for lang, note in resource_notes.items():
            lines.append(f"- **{lang}**: {note}")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
