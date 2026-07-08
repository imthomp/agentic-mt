"""Shared iterative translate -> {estimate -> refine} loop.

Both conditions (self-critique TEaR vs. external-QE-guided) share this
outer loop; they differ only in `step_fn`, which decides whether to freeze
a sentence at this iteration or refine it (see condition_a.py / condition_b.py).
This is the "swap in a different feature/estimate extractor without
rewriting the pipeline" pattern used throughout this project.
"""

import logging
from typing import Callable

import pandas as pd

from agentic_mt.pipelines.llm import chat_turn
from agentic_mt.refinement.extract import extract_translation
from agentic_mt.refinement.prompts import TRANSLATE_TEMPLATE

logger = logging.getLogger(__name__)

# step_fn(source_text, translation, src_lang, tgt_lang, model_name) ->
#   (new_translation, frozen, feedback_str, turns: list[dict])
StepFn = Callable[[str, str, str, str, str], tuple]


def run_iterative(
    test_df: pd.DataFrame,
    step_fn: StepFn,
    condition_name: str,
    src_lang: str,
    tgt_lang: str,
    model_name: str,
    n_iterations: int = 4,
) -> tuple[pd.DataFrame, list[dict]]:
    """Returns (long-format DataFrame with one row per sentence per
    iteration [0..n_iterations], full prompt/response log)."""
    rows = []
    log = []

    for sentence_idx, row in test_df.iterrows():
        source_text = row["source"]
        reference = row["reference"]

        translate_prompt = TRANSLATE_TEMPLATE.format(src_lang=src_lang, tgt_lang=tgt_lang, source_text=source_text)
        raw_response = chat_turn(
            [{"role": "user", "content": translate_prompt}], model_name=model_name, max_new_tokens=256,
        )
        translation = extract_translation(raw_response)
        log.append({"sentence_idx": int(sentence_idx), "condition": condition_name, "iteration": 0,
                    "prompt": translate_prompt, "response": raw_response, "extracted": translation})
        rows.append({"sentence_idx": int(sentence_idx), "condition": condition_name, "iteration": 0,
                     "source": source_text, "reference": reference, "translation": translation, "frozen": False})

        frozen = False
        for it in range(1, n_iterations + 1):
            if frozen:
                rows.append({"sentence_idx": int(sentence_idx), "condition": condition_name, "iteration": it,
                             "source": source_text, "reference": reference, "translation": translation, "frozen": True})
                continue

            translation, frozen, feedback, turns = step_fn(source_text, translation, src_lang, tgt_lang, model_name)
            for t in turns:
                log.append({"sentence_idx": int(sentence_idx), "condition": condition_name, "iteration": it,
                            "feedback": feedback, **t})
            rows.append({"sentence_idx": int(sentence_idx), "condition": condition_name, "iteration": it,
                         "source": source_text, "reference": reference, "translation": translation, "frozen": frozen})

        logger.info(f"{condition_name}: sentence {sentence_idx} done (frozen at iteration "
                    f"{next((r['iteration'] for r in rows if r['sentence_idx']==sentence_idx and r['frozen']), 'never')})")

    return pd.DataFrame(rows), log
