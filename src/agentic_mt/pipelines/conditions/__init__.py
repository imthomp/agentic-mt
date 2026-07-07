"""Translation-condition registry.

Each condition module exposes `run(source_text, src_lang, tgt_lang,
model_name, tm_match) -> dict` with keys `condition`, `final_translation`,
`turns` (full prompt/response log). `tm_match` is unused by conditions that
don't need retrieval (baseline, cot) but kept in the signature so the
orchestrator can call every condition uniformly.
"""

from agentic_mt.pipelines.conditions import l1_baseline, l1_cot, l1_cot_tool, l1_tool

CONDITIONS = {
    "l1_baseline": l1_baseline.run,
    "l1_cot": l1_cot.run,
    "l1_tool": l1_tool.run,
    "l1_cot_tool": l1_cot_tool.run,
}
