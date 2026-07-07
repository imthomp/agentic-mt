"""Logistic-regression triggering models: source-side features -> P(low quality).

Different feature extractors have wildly different row coverage (sentence
length and OOV rate apply to every row; parse depth and domain markers are
English-source only; idiomaticity is whatever subsample was LLM-scored).
Rather than force one row set on every feature, each named model in the
config specifies its own feature list; `fit_model` drops rows missing any
of *that* model's features, so a "universal" model can use the full ~39k
rows while a "full_english_source" model runs on its own smaller, complete
subset. Compare models by AUC + CI, not just coefficient magnitude, since
they're fit on different row sets.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def add_low_quality_label(
    df: pd.DataFrame,
    quality_col: str = "quality_score",
    lang_col: str = "lang",
    quantile: float = 0.25,
) -> pd.Series:
    """Label a row "low quality" if its quality_score falls in the bottom
    `quantile` of its own language's score distribution.

    Per-language (not global) thresholding: Phase 1 showed quality_score
    distributions differ substantially by language, so a single global cutoff
    would just re-encode "which language" rather than "how bad is this row".
    """
    thresholds = df.groupby(lang_col)[quality_col].transform(lambda s: s.quantile(quantile))
    return (df[quality_col] <= thresholds).astype(int)


def fit_model(
    df: pd.DataFrame,
    features: list[str],
    label_col: str = "low_quality",
    n_permutation_repeats: int = 30,
    seed: int = 42,
) -> dict:
    """Fit a standardized logistic regression, report AUC + coefficients +
    permutation importance, on rows with no missing values in `features`."""
    sub = df.dropna(subset=features + [label_col])
    x = sub[features].to_numpy()
    y = sub[label_col].to_numpy()

    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError(f"Label has no variation ({y.sum()}/{len(y)} positive) — cannot fit.")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = LogisticRegression(random_state=seed)
    model.fit(x_scaled, y)

    pred_proba = model.predict_proba(x_scaled)[:, 1]
    auc = roc_auc_score(y, pred_proba)

    perm = permutation_importance(
        model, x_scaled, y, n_repeats=n_permutation_repeats, random_state=seed, scoring="roc_auc",
    )

    importance_df = pd.DataFrame({
        "feature": features,
        "coefficient": model.coef_[0],
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_std": perm.importances_std,
    }).sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)

    return {
        "model": model,
        "scaler": scaler,
        "features": features,
        "n": len(sub),
        "n_positive": int(y.sum()),
        "auc": auc,
        "importance": importance_df,
    }


def save_model(fit_result: dict, path: Path) -> None:
    joblib.dump(
        {"model": fit_result["model"], "scaler": fit_result["scaler"], "features": fit_result["features"]},
        path,
    )
    logger.info(f"Saved model to {path}")
