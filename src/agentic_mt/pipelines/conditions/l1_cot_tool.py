"""L1_cot_tool: full pipeline — model reasons step by step (same 4-step
chain as l1_cot) AND is given the retrieved resource at the decomposition
step, explicitly prompted to decide whether/how to use it."""

from agentic_mt.pipelines.conditions.prompts import DECOMPOSITION_TEMPLATE, DECOMPOSITION_WITH_TOOL_SUFFIX
from agentic_mt.pipelines.conditions.l1_cot import run_chain


def run(source_text: str, src_lang: str, tgt_lang: str, model_name: str, tm_match: dict) -> dict:
    decomposition_prompt = DECOMPOSITION_TEMPLATE.format(
        src_lang=src_lang, tgt_lang=tgt_lang, source_text=source_text,
    ) + DECOMPOSITION_WITH_TOOL_SUFFIX.format(
        tm_source=tm_match["source"], tm_reference=tm_match["reference"],
    )
    return run_chain(
        source_text, src_lang, tgt_lang, model_name, decomposition_prompt, "l1_cot_tool",
        extra_turn_fields={"tm_match": tm_match},
    )
