"""OOV-rate proxy: subword fragmentation under a multilingual tokenizer.

No offline frequency lists exist for Hausa/Khmer/Pashto/Xhosa/Zulu, so we
use the "model's own tokenizer vocabulary" proxy suggested in the brief:
NLLB-200's shared SentencePiece vocabulary. A word that a well-covered
language would split into 1-2 subwords but that fragments into many small
pieces here is a proxy for "rare/unseen for this vocabulary" — the same
intuition behind subword-fragmentation OOV proxies used in low-resource
MT literature.

Caveat: this measures coverage in NLLB's *training* vocabulary, not the
open-source translation model that actually produced each hypothesis in
this dataset (that provenance is heterogeneous and unknown per-row — see
qe/data.py). Treat this as an approximate proxy, not ground-truth OOV.
"""

import pandas as pd

_TOKENIZER = None
FRAGMENTATION_THRESHOLD = 3  # subword pieces beyond this = "OOV-like"


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    return _TOKENIZER


def _oov_rate(sentence: str) -> float:
    words = sentence.split()
    if not words:
        return float("nan")
    tok = _get_tokenizer()
    oov_count = sum(1 for w in words if len(tok.tokenize(w)) > FRAGMENTATION_THRESHOLD)
    return oov_count / len(words)


def extract_oov_rate(df: pd.DataFrame) -> pd.Series:
    return df["source"].fillna("").apply(_oov_rate)
