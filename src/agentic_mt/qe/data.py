"""Load real WMT Direct Assessment human scores and join them with metric
scores already computed by the multilingual-mqm-benchmark project.

Design note: mqmbench's per-metric checkpoint CSVs (scores_cometkiwi.csv,
scores_comet.csv, scores_chrf.csv) do not carry the human quality_score
column, and their segment_id is not unique (many rows share one
lp/year/domain key). Rather than merge on segment_id, we exploit that
pandas boolean filtering preserves row order: filtering the checkpoint CSVs
and a freshly-loaded copy of the same HF dataset by the same `lang` column
both trace back to the same underlying ordered rows, so the two filtered
views line up positionally. `build_merged_dataset` asserts matching
per-language row counts across every source before trusting that alignment.
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

WMT_DA_DATASET = "RicardoRei/wmt-da-human-evaluation"

# Non-English side of a WMT lp code, e.g. "ha-en" -> "ha", "en-ha" -> "ha".
def _lang_from_pair(lp: str) -> str:
    parts = lp.split("-")
    return parts[0] if parts[-1] == "en" else parts[-1]


def load_human_da(target_langs: list[str]) -> pd.DataFrame:
    """Load real WMT DA human scores for the given non-English language codes.

    Returns one row per rated segment: source, hypothesis (MT output already
    submitted by a WMT participant, not generated here), reference, lang,
    da_score (raw z-score), quality_score (per-language min-max normalized
    to [0, 1] for scatter-plot readability only).
    """
    df = load_dataset(WMT_DA_DATASET, split="train").to_pandas()
    df["lang"] = df["lp"].apply(_lang_from_pair)
    df = df[df["lang"].isin(target_langs)].reset_index(drop=True)

    df = df.rename(columns={"src": "source", "mt": "hypothesis", "ref": "reference", "score": "da_score"})

    def _minmax(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return pd.Series(0.5, index=s.index) if hi == lo else (s - lo) / (hi - lo)

    df["quality_score"] = df.groupby("lang")["da_score"].transform(_minmax)

    return df[["source", "hypothesis", "reference", "lang", "domain", "year", "da_score", "quality_score"]]


def _load_metric_checkpoint(mqmbench_root: Path, metric: str, target_langs: list[str]) -> pd.DataFrame:
    path = mqmbench_root / "results" / f"scores_{metric}.csv"
    df = pd.read_csv(path)
    df = df[df["lang"].isin(target_langs)].reset_index(drop=True)
    return df[["lang", metric]]


def build_merged_dataset(
    mqmbench_root: Path,
    target_langs: list[str],
    metrics: list[str] = ("cometkiwi", "comet", "chrf"),
) -> pd.DataFrame:
    """Join freshly-loaded human DA scores with mqmbench's pre-computed metric
    scores for target_langs, by position within each language's row block.

    Raises ValueError if per-language row counts disagree across sources —
    that would mean the positional alignment assumption doesn't hold and the
    merge cannot be trusted.
    """
    human = load_human_da(target_langs)
    human_counts = human["lang"].value_counts().to_dict()

    merged = human.reset_index(drop=True)
    for metric in metrics:
        metric_df = _load_metric_checkpoint(mqmbench_root, metric, target_langs)
        metric_counts = metric_df["lang"].value_counts().to_dict()
        if metric_counts != human_counts:
            raise ValueError(
                f"Row count mismatch for metric '{metric}': "
                f"human={human_counts} vs checkpoint={metric_counts}. "
                "Positional alignment is not safe — refusing to merge."
            )
        if not (metric_df["lang"].values == merged["lang"].values).all():
            raise ValueError(
                f"Row order mismatch for metric '{metric}' — 'lang' columns "
                "differ position-by-position between human and checkpoint data."
            )
        merged[metric] = metric_df[metric].values

    return merged
