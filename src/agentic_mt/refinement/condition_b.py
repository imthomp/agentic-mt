"""Condition B: identical Translate -> {Estimate -> Refine} loop as
condition A, but Estimate is an external QE model (COMET-KIWI) flagging
low-scoring sentences, instead of LLM self-critique.

Standard COMET-KIWI (wmt22-cometkiwi-da) produces one scalar per sentence,
not error spans, so "flag low-scoring spans/sentences" is implemented at
the sentence level here — see results/phase_4_summary.md for why span-level
wasn't available off the shelf.
"""

from agentic_mt.pipelines.llm import chat_turn
from agentic_mt.pipelines.scoring import score_cometkiwi
from agentic_mt.refinement.extract import extract_translation
from agentic_mt.refinement.prompts import QE_REFINE_TEMPLATE

DEFAULT_THRESHOLD = 0.85


def step(
    source_text: str, translation: str, src_lang: str, tgt_lang: str, model_name: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple:
    score = score_cometkiwi([source_text], [translation])[0]
    feedback = f"cometkiwi={score:.4f}"

    if score >= threshold:
        return translation, True, feedback, []

    refine_prompt = QE_REFINE_TEMPLATE.format(
        src_lang=src_lang, tgt_lang=tgt_lang, source_text=source_text, translation=translation, score=score,
    )
    raw_refined = chat_turn([{"role": "user", "content": refine_prompt}], model_name=model_name, max_new_tokens=256)
    refined = extract_translation(raw_refined)
    turns = [{"prompt": refine_prompt, "response": raw_refined, "extracted": refined}]
    return refined, False, feedback, turns
