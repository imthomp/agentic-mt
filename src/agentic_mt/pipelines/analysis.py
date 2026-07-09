"""Paired bootstrap significance tests between conditions, and the core
hypothesis test: does the condition-4-vs-2/3 gap differ between the
high-resource and low-resource pair?
"""

import numpy as np
import pandas as pd

CONDITION_COMPARISONS = [
    ("l1_cot_tool", "l1_cot"),
    ("l1_cot_tool", "l1_tool"),
    ("l1_cot", "l1_baseline"),
    ("l1_tool", "l1_baseline"),
    ("l1_cot_tool", "l1_baseline"),
]


def paired_bootstrap_test(
    scores_a: np.ndarray, scores_b: np.ndarray, n_resamples: int = 1000, seed: int = 42,
) -> dict:
    """Paired bootstrap test for mean(a) - mean(b), resampling sentence
    indices with replacement (same resampled indices applied to both arrays
    since they're the same sentences under different conditions)."""
    assert len(scores_a) == len(scores_b)
    rng = np.random.default_rng(seed)
    n = len(scores_a)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = scores_a[idx].mean() - scores_b[idx].mean()
    observed = scores_a.mean() - scores_b.mean()
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided p-value: proportion of resamples on the other side of zero from the observed diff
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"mean_diff": float(observed), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "p_value": float(p_value)}


def condition_means_table(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Mean score per condition per language pair."""
    return (
        df.groupby(["pair", "condition"])[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def pairwise_significance_table(df: pd.DataFrame, metric: str, n_resamples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Paired bootstrap significance between conditions, per language pair,
    for one metric. Rows are matched by sentence_idx within each pair."""
    rows = []
    for pair, group in df.groupby("pair"):
        wide = group.pivot(index="sentence_idx", columns="condition", values=metric)
        for cond_a, cond_b in CONDITION_COMPARISONS:
            if cond_a not in wide.columns or cond_b not in wide.columns:
                continue
            sub = wide[[cond_a, cond_b]].dropna()
            result = paired_bootstrap_test(sub[cond_a].to_numpy(), sub[cond_b].to_numpy(), n_resamples, seed)
            rows.append({"pair": pair, "metric": metric, "condition_a": cond_a, "condition_b": cond_b, "n": len(sub), **result})
    return pd.DataFrame(rows)


def resource_gap_comparison(
    df: pd.DataFrame, metric: str, high_resource_pair: str, low_resource_pair: str,
    gap_conditions: list[tuple[str, str]] = (("l1_cot_tool", "l1_cot"), ("l1_cot_tool", "l1_tool")),
    n_resamples: int = 1000, seed: int = 42,
) -> pd.DataFrame:
    """The core hypothesis test: for each (cond_a, cond_b) gap, is
    gap(high_resource_pair) significantly different from gap(low_resource_pair)?

    Each language pair's rows are independent samples (different sentences),
    so we bootstrap each pair's gap separately and combine to get a CI on
    the DIFFERENCE of gaps, rather than a paired test.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cond_a, cond_b in gap_conditions:
        gaps = {}
        for pair_name in [high_resource_pair, low_resource_pair]:
            wide = df[df["pair"] == pair_name].pivot(index="sentence_idx", columns="condition", values=metric)
            sub = wide[[cond_a, cond_b]].dropna()
            n = len(sub)
            boot_gaps = np.empty(n_resamples)
            for i in range(n_resamples):
                idx = rng.integers(0, n, size=n)
                boot_gaps[i] = sub[cond_a].to_numpy()[idx].mean() - sub[cond_b].to_numpy()[idx].mean()
            gaps[pair_name] = boot_gaps

        diff_of_gaps = gaps[high_resource_pair] - gaps[low_resource_pair]
        ci_lo, ci_hi = np.percentile(diff_of_gaps, [2.5, 97.5])
        p_value = 2 * min((diff_of_gaps <= 0).mean(), (diff_of_gaps >= 0).mean())
        rows.append({
            "metric": metric,
            "gap": f"{cond_a} - {cond_b}",
            "high_resource_pair": high_resource_pair,
            "low_resource_pair": low_resource_pair,
            "gap_high_resource": float(gaps[high_resource_pair].mean()),
            "gap_low_resource": float(gaps[low_resource_pair].mean()),
            "diff_of_gaps": float(diff_of_gaps.mean()),
            "diff_ci_lo": float(ci_lo),
            "diff_ci_hi": float(ci_hi),
            "p_value": float(p_value),
        })
    return pd.DataFrame(rows)
