"""L1_tool: mechanically prepend a retrieved TM match, single pass, no
reasoning/deliberation prompt — just inject and translate."""

from agentic_mt.pipelines.conditions.prompts import SYSTEM_PROMPT, TOOL_TEMPLATE
from agentic_mt.pipelines.llm import chat_turn


def run(source_text: str, src_lang: str, tgt_lang: str, model_name: str, tm_match: dict) -> dict:
    prompt = TOOL_TEMPLATE.format(
        src_lang=src_lang, tgt_lang=tgt_lang, source_text=source_text,
        tm_source=tm_match["source"], tm_reference=tm_match["reference"],
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    response = chat_turn(messages, model_name=model_name, max_new_tokens=256)
    return {
        "condition": "l1_tool",
        "final_translation": response,
        "turns": [{"prompt": prompt, "response": response}],
        "tm_match": tm_match,
    }
