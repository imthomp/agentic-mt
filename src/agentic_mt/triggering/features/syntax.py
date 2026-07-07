"""Syntactic complexity feature: dependency parse-tree depth.

Per the brief: "otherwise flag as unavailable." spaCy has no trained
dependency-parser pipeline for Hausa, Khmer, Pashto, Xhosa, or Zulu, so
this feature is only computed for English-source rows (the en-ha slice of
the dataset — see triggering/data.py) and is NaN everywhere else. This is
an expected, not a bug: most of this dataset simply cannot be parsed with
currently available open tools.
"""

import pandas as pd

_SPACY_LANG_MODELS = {
    "en": "en_core_web_sm",
}
_NLP_CACHE: dict[str, object] = {}


def _get_nlp(lang: str):
    if lang not in _SPACY_LANG_MODELS:
        return None
    if lang not in _NLP_CACHE:
        import spacy
        try:
            _NLP_CACHE[lang] = spacy.load(_SPACY_LANG_MODELS[lang])
        except OSError:
            _NLP_CACHE[lang] = None
    return _NLP_CACHE[lang]


def _max_depth(token) -> int:
    depth = 0
    node = token
    while node.head != node:
        depth += 1
        node = node.head
    return depth


def extract_parse_depth(df: pd.DataFrame) -> pd.Series:
    result = pd.Series(float("nan"), index=df.index)
    for lang, group in df.groupby("source_lang"):
        nlp = _get_nlp(lang)
        if nlp is None:
            continue
        for idx, text in group["source"].fillna("").items():
            if not text.strip():
                continue
            doc = nlp(text)
            depths = [_max_depth(tok) for tok in doc]
            result.loc[idx] = max(depths) if depths else 0
    return result
