"""Condition A: TEaR's own Translate -> Estimate -> Refine loop (Feng et al.
2025), self-generated 3-shot MQM-style critique as the Estimate step.
"""

from agentic_mt.pipelines.llm import chat_turn
from agentic_mt.refinement.extract import extract_translation
from agentic_mt.refinement.prompts import ESTIMATE_TEMPLATE, REFINE_TEMPLATE


def _has_real_error(mqm_text: str) -> bool:
    """Heuristic matching the paper's own MQM annotation format (Table 18):
    every actual error entry is "category/subcategory - "span""; a segment
    with zero errors has only "critical: no-error" / "major: no-error" /
    "minor: no-error" lines, with no " - " separator anywhere."""
    return " - " in mqm_text


def step(source_text: str, translation: str, src_lang: str, tgt_lang: str, model_name: str) -> tuple:
    estimate_prompt = ESTIMATE_TEMPLATE.format(
        src_lang=src_lang, tgt_lang=tgt_lang, source_text=source_text, translation=translation,
    )
    mqm_annotation = chat_turn(
        [{"role": "user", "content": estimate_prompt}], model_name=model_name, max_new_tokens=256,
    )
    turns = [{"prompt": estimate_prompt, "response": mqm_annotation}]

    if not _has_real_error(mqm_annotation):
        return translation, True, mqm_annotation, turns

    refine_prompt = REFINE_TEMPLATE.format(
        src_lang=src_lang, tgt_lang=tgt_lang, source_text=source_text,
        translation=translation, feedback=mqm_annotation,
    )
    raw_refined = chat_turn([{"role": "user", "content": refine_prompt}], model_name=model_name, max_new_tokens=256)
    refined = extract_translation(raw_refined)
    turns.append({"prompt": refine_prompt, "response": raw_refined, "extracted": refined})
    return refined, False, mqm_annotation, turns
