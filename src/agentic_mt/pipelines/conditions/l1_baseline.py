"""L1_baseline: single-pass "translate this sentence" prompt, no reasoning,
no retrieved resource."""

from agentic_mt.pipelines.conditions.prompts import BASELINE_TEMPLATE, SYSTEM_PROMPT
from agentic_mt.pipelines.llm import chat_turn


def run(source_text: str, src_lang: str, tgt_lang: str, model_name: str, tm_match: dict | None = None) -> dict:
    prompt = BASELINE_TEMPLATE.format(src_lang=src_lang, tgt_lang=tgt_lang, source_text=source_text)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    response = chat_turn(messages, model_name=model_name, max_new_tokens=256)
    return {
        "condition": "l1_baseline",
        "final_translation": response,
        "turns": [{"prompt": prompt, "response": response}],
    }
