"""Feature-extractor registry.

Every extractor is a callable `extract(df: pd.DataFrame) -> pd.Series`
taking the triggering dataset (must have at least `source` and
`source_lang` columns) and returning one score per row, aligned to df's
index. Extractors that can't handle a row's source language return NaN for
that row rather than raising — missingness is a first-class signal here,
not an error.

To add a new feature extractor: write the function elsewhere, then
register it in FEATURE_EXTRACTORS below. The pipeline (pipeline.py) never
needs to change.
"""

from agentic_mt.triggering.features.domain import extract_domain_marker_count
from agentic_mt.triggering.features.idiomaticity import extract_idiomaticity
from agentic_mt.triggering.features.length import extract_sentence_length
from agentic_mt.triggering.features.oov import extract_oov_rate
from agentic_mt.triggering.features.syntax import extract_parse_depth

FEATURE_EXTRACTORS = {
    "sentence_length": extract_sentence_length,
    "oov_rate": extract_oov_rate,
    "idiomaticity": extract_idiomaticity,
    "parse_depth": extract_parse_depth,
    "domain_marker_count": extract_domain_marker_count,
}
