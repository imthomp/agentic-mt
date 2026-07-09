"""Source/reference pairs for the two language-pair conditions.

Both pairs use English as the source, sourced from real human-evaluated
WMT data already used in Phases 1/2, so scoring against real references is
possible:

- en-de: WMT MQM (RicardoRei/wmt-mqm-human-evaluation) — high-resource
  replication check against Wu, Aycock & Monz (2025).
- en-ha: WMT DA (RicardoRei/wmt-da-human-evaluation), Phase 1's low-resource
  pair, restricted to the en-ha direction (see Phase 2's triggering/data.py
  for why direction matters here).

Each pair is split into a small test sample (what we actually translate)
and a much larger "TM corpus" (everything else) used as the retrieval pool
for the tool-use conditions — so a test sentence is never trivially
retrieved against itself.
"""

import json
import functools

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from agentic_mt.qe.data import load_human_da

WMT_MQM_DATASET = "RicardoRei/wmt-mqm-human-evaluation"
FLORES_DATASET = "openlanguagedata/flores_plus"


def load_ende_pairs() -> pd.DataFrame:
    df = load_dataset(WMT_MQM_DATASET, split="train").to_pandas()
    ende = df[df["lp"] == "en-de"][["src", "ref"]].drop_duplicates().reset_index(drop=True)
    return ende.rename(columns={"src": "source", "ref": "reference"})


def load_enha_pairs() -> pd.DataFrame:
    df = load_human_da(["ha"])
    enha = df[df["lp"] == "en-ha"][["source", "reference"]].drop_duplicates().reset_index(drop=True)
    return enha


def _load_flores_lang(flores_code: str, split: str = "devtest") -> pd.DataFrame:
    path = hf_hub_download(FLORES_DATASET, f"{split}/{flores_code}.jsonl", repo_type="dataset")
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return pd.DataFrame(rows)[["id", "text"]]


def load_flores_pair(tgt_flores_code: str) -> pd.DataFrame:
    """English-source pairs from FLORES-plus devtest, joined by sentence id.

    Used for language directions absent from the WMT DA/MQM data entirely
    (e.g. en-xh, en-zu, en-sw all have zero rows in wmt-da-human-evaluation —
    xh/zu only appear as xh-zu/zu-xh there, never with English). No human
    quality scores here, but Phase 3+'s design doesn't need them — only
    (source, reference) pairs to translate and score with COMET/chrF.
    """
    eng = _load_flores_lang("eng_Latn").rename(columns={"text": "source"})
    tgt = _load_flores_lang(tgt_flores_code).rename(columns={"text": "reference"})
    merged = eng.merge(tgt, on="id")[["source", "reference"]]
    return merged.drop_duplicates(subset="source").reset_index(drop=True)


LOADERS = {
    "en-de": load_ende_pairs,
    "en-ha": load_enha_pairs,
    "en-xh": functools.partial(load_flores_pair, "xho_Latn"),
    "en-zu": functools.partial(load_flores_pair, "zul_Latn"),
    "en-sw": functools.partial(load_flores_pair, "swh_Latn"),
}


def sample_test_and_tm_corpus(
    pair: str, n_test: int, seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (test_df, tm_corpus_df) for the given pair name ("en-de"/"en-ha")."""
    df = LOADERS[pair]()
    # Some sources have multiple reference variants (e.g. en-de MQM has
    # several post-edited references per segment) — dedupe to one row per
    # source first, so a source sentence can never appear in both the test
    # sample and the TM corpus under a different reference.
    df = df.drop_duplicates(subset="source").reset_index(drop=True)
    test_df = df.sample(n=n_test, random_state=seed).reset_index(drop=True)
    tm_corpus_df = df[~df["source"].isin(test_df["source"])].reset_index(drop=True)
    return test_df, tm_corpus_df
