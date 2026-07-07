"""Correlation statistics with bootstrap confidence intervals."""

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    stat_fn,
    n_resamples: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI for stat_fn(x, y), resampling paired (x, y) rows."""
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_stats[i] = stat_fn(x[idx], y[idx])
    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100
    return float(np.percentile(boot_stats, lo_pct)), float(np.percentile(boot_stats, hi_pct))


def compute_correlations(
    metric_score: np.ndarray,
    human_score: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Pearson and Spearman correlation between a metric score and human score,
    with 95% bootstrap confidence intervals (percentile method)."""
    pearson_r, pearson_p = stats.pearsonr(metric_score, human_score)
    spearman_r, spearman_p = stats.spearmanr(metric_score, human_score)

    pearson_lo, pearson_hi = bootstrap_ci(
        metric_score, human_score,
        stat_fn=lambda x, y: stats.pearsonr(x, y).statistic,
        n_resamples=n_resamples, seed=seed,
    )
    spearman_lo, spearman_hi = bootstrap_ci(
        metric_score, human_score,
        stat_fn=lambda x, y: stats.spearmanr(x, y).statistic,
        n_resamples=n_resamples, seed=seed,
    )

    return {
        "n": len(metric_score),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "pearson_ci_lo": pearson_lo,
        "pearson_ci_hi": pearson_hi,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "spearman_ci_lo": spearman_lo,
        "spearman_ci_hi": spearman_hi,
    }
