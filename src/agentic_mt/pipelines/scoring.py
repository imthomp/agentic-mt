"""Score condition outputs with COMET-KIWI (reference-free), reference-based
COMET, and chrF. Small-scale (tens to low hundreds of sentences per run), so
no chunking/caching complexity is needed here (contrast with mqmbench's
comet.py, built for tens of thousands of segments).
"""

import sacrebleu
from comet import download_model, load_from_checkpoint

_MODEL_CACHE: dict[str, object] = {}


def _load_comet(model_name: str):
    if model_name not in _MODEL_CACHE:
        path = download_model(model_name)
        _MODEL_CACHE[model_name] = load_from_checkpoint(path)
    return _MODEL_CACHE[model_name]


def score_cometkiwi(
    sources: list[str], hypotheses: list[str], model_name: str = "Unbabel/wmt22-cometkiwi-da", gpus: int = 1,
) -> list[float]:
    model = _load_comet(model_name)
    data = [{"src": s, "mt": h} for s, h in zip(sources, hypotheses)]
    return model.predict(data, batch_size=16, gpus=gpus).scores


def score_comet(
    sources: list[str], hypotheses: list[str], references: list[str],
    model_name: str = "Unbabel/wmt22-comet-da", gpus: int = 1,
) -> list[float]:
    model = _load_comet(model_name)
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
    return model.predict(data, batch_size=16, gpus=gpus).scores


def score_chrf(hypotheses: list[str], references: list[str]) -> list[float]:
    chrf = sacrebleu.CHRF()
    return [chrf.sentence_score(h, [r]).score for h, r in zip(hypotheses, references)]


def score_all(sources: list[str], hypotheses: list[str], references: list[str], gpus: int = 1) -> dict[str, list[float]]:
    return {
        "cometkiwi": score_cometkiwi(sources, hypotheses, gpus=gpus),
        "comet": score_comet(sources, hypotheses, references, gpus=gpus),
        "chrf": score_chrf(hypotheses, references),
    }
