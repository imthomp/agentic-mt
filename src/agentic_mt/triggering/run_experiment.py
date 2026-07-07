"""Triggering-feature experiment: which source-side features predict low
translation quality for low-resource pairs?

Usage:
    uv run python -m agentic_mt.triggering.run_experiment --config configs/triggering_low_resource.yaml
"""

import argparse
import copy
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from scipy import stats

from agentic_mt.triggering.data import load_triggering_dataset
from agentic_mt.triggering.modeling import add_low_quality_label, fit_model, save_model
from agentic_mt.triggering.pipeline import extract_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

ALL_FEATURES = ["sentence_length", "oov_rate", "idiomaticity", "parse_depth", "domain_marker_count"]


def _univariate_table(df: pd.DataFrame, features: list[str], label_col: str, quality_col: str) -> pd.DataFrame:
    rows = []
    for feat in features:
        sub = df.dropna(subset=[feat, quality_col])
        if len(sub) < 10 or sub[label_col].nunique() < 2:
            rows.append({"feature": feat, "n": len(sub), "pearson_r": float("nan"),
                         "spearman_r": float("nan"), "univariate_auc": float("nan")})
            continue
        pearson_r, _ = stats.pearsonr(sub[feat], sub[quality_col])
        spearman_r, _ = stats.spearmanr(sub[feat], sub[quality_col])

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        x = StandardScaler().fit_transform(sub[[feat]])
        y = sub[label_col].to_numpy()
        model = LogisticRegression().fit(x, y)
        auc = roc_auc_score(y, model.predict_proba(x)[:, 1])

        rows.append({"feature": feat, "n": len(sub), "pearson_r": pearson_r,
                     "spearman_r": spearman_r, "univariate_auc": auc})
    return pd.DataFrame(rows).sort_values("univariate_auc", ascending=False).reset_index(drop=True)


def _univariate_table_by_lang(df: pd.DataFrame, features: list[str], label_col: str, quality_col: str) -> pd.DataFrame:
    """Per-language breakdown of the same univariate stats.

    quality_score is min-max normalized *within* each language (Phase 1), so
    pooling rows across languages for one global correlation risks conflating
    genuine feature-quality relationships with between-language baseline
    differences (a Simpson's-paradox-shaped risk). This table is the
    within-language ground truth; the pooled table above is a coarser
    summary, not a replacement for it.
    """
    frames = []
    for lang, group in df.groupby("lang"):
        t = _univariate_table(group, features, label_col, quality_col)
        t.insert(0, "lang", lang)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def extract_lang_features(config: dict, lang: str) -> Path:
    """Array-task entry point: extract every configured feature for a single
    language's rows and write a stable (non-timestamped) per-language CSV.

    Runs the full row set for `lang` — no LLM subsampling — so this is meant
    for a SLURM array task (one GPU per language), not the login node.
    """
    feature_names = config.get("feature_extractors", ALL_FEATURES)
    extractor_kwargs = copy.deepcopy(config.get("extractor_kwargs", {}))

    # Give each array task its own cache file so concurrent per-language jobs
    # never write to the same path.
    for kwargs in extractor_kwargs.values():
        if "cache_file" in kwargs:
            p = Path(kwargs["cache_file"])
            kwargs["cache_file"] = str(p.with_name(f"{p.stem}_{lang}{p.suffix}"))

    features_dir = Path(config["output_dir"]).expanduser() / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading triggering dataset for lang={lang}")
    df = load_triggering_dataset([lang])

    logger.info(f"Extracting features for {lang}: {feature_names}")
    df = extract_features(df, feature_names, extractor_kwargs)

    out_path = features_dir / f"features_{lang}.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote {lang} feature table to {out_path}")
    return out_path


def run(config: dict) -> Path:
    target_langs = config["target_langs"]
    feature_names = config.get("feature_extractors", ALL_FEATURES)
    extractor_kwargs = config.get("extractor_kwargs", {})
    label_quantile = config.get("low_quality_quantile", 0.25)
    seed = config["random_seed"]
    model_specs = config["models"]

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = Path(config["output_dir"]).expanduser() / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    features_dir = Path(config["output_dir"]).expanduser() / "features"
    per_lang_paths = [features_dir / f"features_{lang}.csv" for lang in target_langs]
    if all(p.exists() for p in per_lang_paths):
        logger.info(f"Loading precomputed per-language feature tables from {features_dir}")
        df = pd.concat([pd.read_csv(p) for p in per_lang_paths], ignore_index=True)
    else:
        logger.info(f"Loading triggering dataset for {target_langs}")
        df = load_triggering_dataset(target_langs)
        logger.info(f"Extracting features: {feature_names}")
        df = extract_features(df, feature_names, extractor_kwargs)

    df["low_quality"] = add_low_quality_label(df, quantile=label_quantile)

    features_path = out_dir / "features.csv"
    df.to_csv(features_path, index=False)
    logger.info(f"Wrote feature table to {features_path}")

    univariate = _univariate_table(df, feature_names, "low_quality", "quality_score")
    univariate_path = out_dir / "univariate_correlations.csv"
    univariate.to_csv(univariate_path, index=False)

    univariate_by_lang = _univariate_table_by_lang(df, feature_names, "low_quality", "quality_score")
    univariate_by_lang_path = out_dir / "univariate_correlations_by_lang.csv"
    univariate_by_lang.to_csv(univariate_by_lang_path, index=False)

    model_results = {}
    for spec in model_specs:
        name = spec["name"]
        features = spec["features"]
        row_filter = spec.get("row_filter")
        sub_df = df.query(row_filter) if row_filter else df

        try:
            result = fit_model(sub_df, features, seed=seed)
        except ValueError as e:
            logger.warning(f"Skipping model '{name}': {e}")
            continue

        model_results[name] = result
        save_model(result, out_dir / f"model_{name}.joblib")
        result["importance"].to_csv(out_dir / f"importance_{name}.csv", index=False)
        logger.info(f"Model '{name}': AUC={result['auc']:.3f} n={result['n']} ({result['n_positive']} positive)")

    _plot_importance(model_results, out_dir / "feature_importance.png")

    report_path = out_dir / "report.md"
    _write_report(report_path, univariate, univariate_by_lang, model_results, target_langs, label_quantile)

    print(f"Wrote feature table to {features_path}")
    print(f"Wrote univariate correlations to {univariate_path}")
    print(f"Wrote report to {report_path}")
    return out_dir


def _plot_importance(model_results: dict, out_path: Path) -> None:
    n_models = len(model_results)
    if n_models == 0:
        return
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for ax, (name, result) in zip(axes, model_results.items()):
        imp = result["importance"]
        ax.barh(imp["feature"], imp["permutation_importance_mean"],
                xerr=imp["permutation_importance_std"])
        ax.set_title(f"{name} (AUC={result['auc']:.3f}, n={result['n']})")
        ax.set_xlabel("permutation importance (Δ AUC)")
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_report(path: Path, univariate: pd.DataFrame, univariate_by_lang: pd.DataFrame, model_results: dict,
                   target_langs: list[str], label_quantile: float) -> None:
    lines = ["# Triggering-feature experiment\n"]
    lines.append(f"Languages: {target_langs}. Low-quality label = bottom "
                 f"{label_quantile:.0%} of quality_score, per language.\n")
    lines.append("## Univariate: feature vs. quality_score / low-quality label (pooled across languages)\n")
    lines.append(univariate.to_markdown(index=False))
    lines.append("\n## Univariate, by language (see caveat in module docstring re: pooling)\n")
    lines.append(univariate_by_lang.to_markdown(index=False))
    lines.append("\n## Multivariate logistic regression models\n")
    for name, result in model_results.items():
        lines.append(f"\n### {name} (AUC={result['auc']:.3f}, n={result['n']}, "
                     f"{result['n_positive']} positive)\n")
        lines.append(result["importance"].to_markdown(index=False))
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--lang", default=None,
                        help="If set, only extract features for this language and exit "
                             "(SLURM array-task mode) instead of running the full merge+model step.")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.lang:
        extract_lang_features(cfg, args.lang)
    else:
        run(cfg)
