"""Load the triggering-feature dataset: Phase 1's WMT DA rows, with the
per-row source language exposed (Phase 1 collapses direction into a single
non-English `lang` code; source-side features need to know which side of
the pair is actually the source sentence).
"""

from pathlib import Path

import pandas as pd

from agentic_mt.qe.data import load_human_da


def load_triggering_dataset(target_langs: list[str]) -> pd.DataFrame:
    """Return Phase 1's rows plus a `source_lang` column (e.g. "en" or "ha").

    Per WMT convention `lp` is "src-tgt", so the source language is the
    first component.
    """
    df = load_human_da(target_langs)
    df["source_lang"] = df["lp"].str.split("-").str[0]
    return df
