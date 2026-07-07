"""FAISS retrieval over a held-out parallel corpus — the "TM you have
access to" for the tool-use conditions (L1_tool, L1_cot_tool).

Uses LaBSE, a sentence embedding model trained specifically for
cross-lingual bitext mining/retrieval (109 languages, including Hausa),
so the same approach works for both the high- and low-resource pair.
"""

import numpy as np
import pandas as pd

_ENCODER = None


def _get_encoder(device: str | None = None):
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _ENCODER = SentenceTransformer("sentence-transformers/LaBSE", device=device)
    return _ENCODER


class TMIndex:
    """A FAISS index over a corpus's source-side sentences, for retrieving
    the nearest TM match (source, reference) pair to a new query sentence."""

    def __init__(self, corpus: pd.DataFrame, device: str | None = None):
        import faiss

        self.corpus = corpus.reset_index(drop=True)
        self.device = device
        encoder = _get_encoder(device)
        embeddings = encoder.encode(
            self.corpus["source"].tolist(), normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query: str, k: int = 1) -> list[dict]:
        encoder = _get_encoder(self.device)
        q_emb = encoder.encode([query], normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        scores, idx = self.index.search(q_emb, k)
        matches = []
        for score, i in zip(scores[0], idx[0]):
            row = self.corpus.iloc[int(i)]
            matches.append({"source": row["source"], "reference": row["reference"], "similarity": float(score)})
        return matches
