"""Phase 3 addendum, part 3: format-learning pre-test for a redesigned
CoT condition on Aya-101.

Phase 3's CoT confound wasn't just bad prompting -- chain-of-thought as
originally conceived assumes reasoning tokens and source tokens share one
self-attention stream, which holds for decoder-only chat models but not
for Aya-101's encoder-decoder architecture (the decoder only sees the
source through cross-attention to a representation the encoder computed
before any "reasoning" happens). Before designing a new unconfounded CoT
condition around this constraint, this checks the precondition that
interpretation depends on: can Aya-101 even produce coherent structured
output (source-specific analysis, distinct from the translation) at all?

If it can't, a null result in any redesigned condition is uninterpretable
-- it would be impossible to tell "the theory doesn't hold" from "the
model can't execute the strategy," which is exactly the ambiguity that
made Phase 3's CoT chain unusable in the first place.

Few-shot (3 examples), single pass, en-ha, small n (this is a precondition
check, not a quality measurement).
"""
import json
import logging
from pathlib import Path

from agentic_mt.pipelines.data import sample_test_and_tm_corpus
from agentic_mt.pipelines.llm import chat_turn, unload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "CohereForAI/aya-101"
PAIR = "en-ha"
TGT_LANG = "Hausa"
N_TEST = 20
SEED = 43  # different from Phase 3's seed=42 -- these must be held-out sentences, not reused as few-shot demonstrations

SYSTEM_PROMPT = "You are a helpful assistant."

FEWSHOT = """Translate the following English sentences into Hausa. For each one, first give a short Analysis of anything tricky about the sentence (idioms, ambiguous words, cultural references), then give the Translation.

English: The early bird catches the worm.
Analysis: This is an English idiom about the advantage of acting before others. A literal translation would not make sense in Hausa; it needs an equivalent expression or a plain restatement of the meaning.
Translation: Wanda ya tashi da wuri shi ne yake samun galaba.

English: She kicked the bucket last night.
Analysis: "Kicked the bucket" is an informal idiom meaning "died." A literal translation would be nonsensical and inappropriate; the plain meaning should be translated instead.
Translation: Ta mutu jiya dare.

English: The bank was closed on Sunday.
Analysis: "Bank" is ambiguous in English (financial institution vs. riverbank), but context (closed on Sunday) makes it clearly the financial institution.
Translation: An rufe banki ranar Lahadi.

English: {source_text}
Analysis:"""


def is_coherent(response: str) -> dict:
    """Cheap heuristic checks, not a quality judgment -- just: did the model
    produce the two-part structure at all, and is the "translation" part
    actually different text from the "analysis" part (not a repeat)."""
    has_translation_marker = "Translation:" in response
    if not has_translation_marker:
        return {"has_structure": False, "analysis": response.strip(), "translation": "", "degenerate_repeat": None}
    analysis, _, translation = response.partition("Translation:")
    analysis, translation = analysis.strip(), translation.strip()
    non_empty = len(analysis) > 0 and len(translation) > 0
    degenerate_repeat = analysis.strip() == translation.strip() if non_empty else None
    return {
        "has_structure": has_translation_marker and non_empty,
        "analysis": analysis, "translation": translation,
        "degenerate_repeat": degenerate_repeat,
    }


test_df, _ = sample_test_and_tm_corpus(PAIR, N_TEST, SEED)

rows = []
for i, (_, row) in enumerate(test_df.iterrows()):
    prompt = FEWSHOT.format(source_text=row["source"])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    response = chat_turn(messages, model_name=MODEL_NAME, max_new_tokens=300)
    checks = is_coherent(response)
    rows.append({"source": row["source"], "raw_response": response, **checks})
    logger.info(f"{i+1}/{N_TEST}: has_structure={checks['has_structure']}")

unload()

out_path = Path("results/phase3_pretest_format_enha.jsonl")
with out_path.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

n_structured = sum(r["has_structure"] for r in rows)
n_degenerate = sum(1 for r in rows if r["degenerate_repeat"])
result = {
    "pair": PAIR, "model": MODEL_NAME, "n_sentences": N_TEST,
    "n_structured": n_structured,
    "n_degenerate_repeat": n_degenerate,
    "pretest_passes": n_structured >= N_TEST * 0.8,  # arbitrary but pre-specified threshold
}
out_scores_path = Path("results/phase3_pretest_format_enha_result.json")
with out_scores_path.open("w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
print(f"Wrote {len(rows)} rows to {out_path}, result to {out_scores_path}")
