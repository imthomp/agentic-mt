"""Run a configurable set of feature extractors over the triggering dataset.

Modularity note: this module never names a specific extractor. Add/remove
extractors by editing FEATURE_EXTRACTORS in features/__init__.py and the
`feature_extractors` list in the experiment config — nothing here changes.
"""

import logging

import pandas as pd

from agentic_mt.triggering.features import FEATURE_EXTRACTORS

logger = logging.getLogger(__name__)


def extract_features(
    df: pd.DataFrame,
    feature_names: list[str],
    extractor_kwargs: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Run each named extractor and attach its output as a column on a copy of df."""
    extractor_kwargs = extractor_kwargs or {}
    out = df.copy()
    for name in feature_names:
        if name not in FEATURE_EXTRACTORS:
            raise KeyError(f"Unknown feature extractor '{name}'. Available: {list(FEATURE_EXTRACTORS)}")
        logger.info(f"Extracting feature: {name}")
        fn = FEATURE_EXTRACTORS[name]
        kwargs = extractor_kwargs.get(name, {})
        out[name] = fn(out, **kwargs)
        n_available = out[name].notna().sum()
        logger.info(f"  {name}: {n_available}/{len(out)} rows available")
    return out
