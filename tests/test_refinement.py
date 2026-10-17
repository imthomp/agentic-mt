from agentic_mt.refinement import condition_b
from agentic_mt.refinement.extract import extract_translation


def test_extract_translation_removes_preamble_and_label():
    response = "The revised translation would be:\nTranslation: Guten Morgen"
    assert extract_translation(response) == "Guten Morgen"


def test_qe_step_freezes_translation_above_threshold(monkeypatch):
    monkeypatch.setattr(condition_b, "score_cometkiwi", lambda sources, translations: [0.91])
    translation, frozen, feedback, turns = condition_b.step(
        "source", "translation", "English", "German", "model"
    )
    assert translation == "translation"
    assert frozen is True
    assert feedback == "cometkiwi=0.9100"
    assert turns == []


def test_qe_step_refines_translation_below_threshold(monkeypatch):
    monkeypatch.setattr(condition_b, "score_cometkiwi", lambda sources, translations: [0.2])
    monkeypatch.setattr(condition_b, "chat_turn", lambda *args, **kwargs: "Translation: verbessert")
    translation, frozen, feedback, turns = condition_b.step(
        "source", "draft", "English", "German", "model"
    )
    assert translation == "verbessert"
    assert frozen is False
    assert feedback == "cometkiwi=0.2000"
    assert len(turns) == 1
