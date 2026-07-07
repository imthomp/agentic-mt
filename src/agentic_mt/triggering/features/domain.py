"""Domain-marker feature: count of technical/medical/legal keywords.

Same availability caveat as syntax.py: we only have curated term lists for
English, so this is computed only for English-source rows and NaN
elsewhere ("flag as unavailable", per the brief, extended by analogy from
the syntactic-complexity requirement to this feature too — there's no
offline domain lexicon for Hausa/Khmer/Pashto/Xhosa/Zulu either).

The term lists are intentionally small and illustrative, not exhaustive —
swap in a real curated lexicon by replacing DOMAIN_TERMS.
"""

import re

import pandas as pd

DOMAIN_TERMS = {
    "medical": {
        "patient", "diagnosis", "treatment", "symptom", "disease", "clinical",
        "medication", "dosage", "surgery", "vaccine", "infection", "therapy",
        "prescription", "diagnosed", "syndrome", "chronic", "acute", "tumor",
    },
    "legal": {
        "plaintiff", "defendant", "statute", "jurisdiction", "litigation",
        "contract", "liability", "clause", "tribunal", "appeal", "verdict",
        "testimony", "counsel", "prosecution", "regulation", "compliance",
    },
    "technical": {
        "algorithm", "protocol", "server", "database", "encryption", "bandwidth",
        "compiler", "firmware", "architecture", "latency", "throughput",
        "api", "kernel", "framework", "runtime", "deploy", "configuration",
    },
}

_ALL_TERMS = {term for terms in DOMAIN_TERMS.values() for term in terms}
_WORD_RE = re.compile(r"[a-zA-Z]+")


def _count_markers(text: str) -> int:
    words = {w.lower() for w in _WORD_RE.findall(text)}
    return len(words & _ALL_TERMS)


def extract_domain_marker_count(df: pd.DataFrame) -> pd.Series:
    result = pd.Series(float("nan"), index=df.index)
    en_mask = df["source_lang"] == "en"
    if en_mask.any():
        result.loc[en_mask] = df.loc[en_mask, "source"].fillna("").apply(_count_markers).astype(float)
    return result
