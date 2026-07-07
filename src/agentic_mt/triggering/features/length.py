"""Sentence-length feature: whitespace token count.

Limitation: Khmer (km) is written without spaces between words, so
whitespace tokenization badly undercounts "tokens" for km rows (it
effectively measures clause/punctuation-delimited chunks, not words). We
compute it anyway for cross-language consistency but this is flagged in
results/phase_2_summary.md — do not treat km sentence_length as
comparable to the other languages'.
"""

import pandas as pd


def extract_sentence_length(df: pd.DataFrame) -> pd.Series:
    return df["source"].fillna("").str.split().apply(len)
