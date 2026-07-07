"""L1_cot: decompose/draft/refine/proofread prompt chain, no external
resource — replicates Wu, Aycock & Monz (2025)'s step-by-step baseline
(Figures 9-12) as closely as possible from their paper's method description.
Each step is a new turn in the same conversation, with full prior context
retained (matching the paper's stated setup, Section 2)."""

from agentic_mt.pipelines.conditions.prompts import (
    DECOMPOSITION_TEMPLATE,
    DRAFT_TEMPLATE,
    PROOFREADING_TEMPLATE,
    REFINEMENT_TEMPLATE,
    SYSTEM_PROMPT,
)
from agentic_mt.pipelines.llm import chat_turn


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text.strip() else text


def run_chain(
    source_text: str,
    src_lang: str,
    tgt_lang: str,
    model_name: str,
    decomposition_prompt: str,
    condition_name: str,
    extra_turn_fields: dict | None = None,
) -> dict:
    """Shared 4-step chain; only the decomposition prompt differs between
    l1_cot (no tool) and l1_cot_tool (tool-augmented decomposition)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    turns = []

    messages.append({"role": "user", "content": decomposition_prompt})
    decomposition = chat_turn(messages, model_name=model_name, max_new_tokens=512)
    messages.append({"role": "assistant", "content": decomposition})
    turns.append({"prompt": decomposition_prompt, "response": decomposition})

    draft_prompt = DRAFT_TEMPLATE.format(src_lang=src_lang, tgt_lang=tgt_lang, source_text=source_text)
    messages.append({"role": "user", "content": draft_prompt})
    draft = chat_turn(messages, model_name=model_name, max_new_tokens=256)
    messages.append({"role": "assistant", "content": draft})
    turns.append({"prompt": draft_prompt, "response": draft})

    refine_prompt = REFINEMENT_TEMPLATE
    messages.append({"role": "user", "content": refine_prompt})
    refined = chat_turn(messages, model_name=model_name, max_new_tokens=256)
    messages.append({"role": "assistant", "content": refined})
    turns.append({"prompt": refine_prompt, "response": refined})

    proofread_prompt = PROOFREADING_TEMPLATE.format(source_text=source_text, draft=draft, refined=refined)
    messages.append({"role": "user", "content": proofread_prompt})
    proofread = chat_turn(messages, model_name=model_name, max_new_tokens=256)
    turns.append({"prompt": proofread_prompt, "response": proofread})

    result = {
        "condition": condition_name,
        "final_translation": _first_line(proofread),
        "turns": turns,
    }
    if extra_turn_fields:
        result.update(extra_turn_fields)
    return result


def run(source_text: str, src_lang: str, tgt_lang: str, model_name: str, tm_match: dict | None = None) -> dict:
    decomposition_prompt = DECOMPOSITION_TEMPLATE.format(
        src_lang=src_lang, tgt_lang=tgt_lang, source_text=source_text,
    )
    return run_chain(source_text, src_lang, tgt_lang, model_name, decomposition_prompt, "l1_cot")
