"""Per-iteration score means, peak-iteration detection, and the plot."""

import matplotlib.pyplot as plt
import pandas as pd


def iteration_means(scored_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    return scored_df.groupby(["condition", "iteration"])[metrics].mean().reset_index()


def peak_analysis(means_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """iteration-of-peak and total gain from iteration 1 to peak, per
    condition — computed over iterations 1..N only (excludes iteration 0,
    the pre-refinement initial translation), per the brief."""
    rows = []
    for condition, group in means_df.groupby("condition"):
        post_it = group[group["iteration"] >= 1]
        peak_row = post_it.loc[post_it[metric].idxmax()]
        iter1_score = post_it[post_it["iteration"] == 1][metric].iloc[0]
        rows.append({
            "condition": condition,
            "metric": metric,
            "score_iter1": float(iter1_score),
            "iteration_of_peak": int(peak_row["iteration"]),
            "score_at_peak": float(peak_row[metric]),
            "gain_iter1_to_peak": float(peak_row[metric] - iter1_score),
        })
    return pd.DataFrame(rows)


def plot_quality_vs_iteration(means_df: pd.DataFrame, metrics: list[str], out_path) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        for condition, group in means_df.groupby("condition"):
            group = group.sort_values("iteration")
            ax.plot(group["iteration"], group[metric], marker="o", label=condition)
        ax.set_xlabel("iteration (0 = initial translation)")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
