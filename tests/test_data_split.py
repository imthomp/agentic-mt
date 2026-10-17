import pandas as pd

from agentic_mt.pipelines import data


def test_test_and_retrieval_corpora_do_not_share_sources(monkeypatch):
    frame = pd.DataFrame(
        {
            "source": ["duplicate", "duplicate", "one", "two", "three"],
            "reference": ["a", "b", "c", "d", "e"],
        }
    )
    monkeypatch.setitem(data.LOADERS, "test-pair", lambda: frame)

    test, corpus = data.sample_test_and_tm_corpus("test-pair", n_test=2, seed=7)

    assert len(test) == 2
    assert set(test["source"]).isdisjoint(corpus["source"])
    assert len(test) + len(corpus) == 4


def test_sampling_is_deterministic(monkeypatch):
    frame = pd.DataFrame(
        {"source": [str(i) for i in range(10)], "reference": [str(i) for i in range(10)]}
    )
    monkeypatch.setitem(data.LOADERS, "test-pair", lambda: frame)
    first, _ = data.sample_test_and_tm_corpus("test-pair", n_test=3, seed=42)
    second, _ = data.sample_test_and_tm_corpus("test-pair", n_test=3, seed=42)
    assert first.equals(second)
