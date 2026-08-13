"""Phase 3 addendum, part 2 (revised): best-of-N Level-1 baseline on en-de.

Originally planned to rerun best-of-N on Khmer (Phase 1's best-correlated
low-resource pair, r=0.457) as a QE-trustworthy comparison to the en-ha
run. Turns out that number is measured in the km-en direction (Khmer
source, English hypothesis) -- Phase 1's WMT DA data has no en-km rows at
all. Same problem for ps (ps-en only) and xh/zu (xh-zu/zu-xh only, never
against English). Only Hausa has real en-ha human-judgment data, and
that's the pair already flagged as unreliable (0.115) -- there is no
QE-trustworthy low-resource fallback pair in this direction. Worth its own
line in the report: direction matters, and we don't have a clean one to
fall back on for the low-resource case.

Rerouted to en-de instead: not low-resource, but it's the one pair with
both (a) a genuinely directional (en-de) QE correlation on record (~0.57,
mqmbench) and (b) an existing Phase 3 four-condition baseline to compare
against. Tests whether "best-of-N beats every architected condition" is
specific to en-ha's broken QE judge, or holds even when the judge is
trustworthy.
"""
import json
import logging
from pathlib import Path

from agentic_mt.pipelines.conditions.prompts import BASELINE_TEMPLATE, SYSTEM_PROMPT
from agentic_mt.pipelines.data import sample_test_and_tm_corpus
from agentic_mt.pipelines.llm import sample_n, unload
from agentic_mt.pipelines.scoring import score_cometkiwi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "CohereForAI/aya-101"
PAIR = "en-de"
TGT_LANG = "German"
N = 5
N_TEST = 100
SEED = 42

test_df, _ = sample_test_and_tm_corpus(PAIR, N_TEST, SEED)

rows = []
for i, (_, row) in enumerate(test_df.iterrows()):
    prompt = BASELINE_TEMPLATE.format(src_lang="English", tgt_lang=TGT_LANG, source_text=row["source"])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    samples = sample_n(messages, MODEL_NAME, n=N, temperature=0.7, max_new_tokens=256)
    rows.append({"source": row["source"], "reference": row["reference"], "samples": samples})
    logger.info(f"{i+1}/{N_TEST}: done")

unload()

out_path = Path("results/phase3_bestofn_ende.jsonl")
with out_path.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
logger.info(f"Wrote {len(rows)} rows to {out_path}")

sources_flat, samples_flat = [], []
for r in rows:
    for s in r["samples"]:
        sources_flat.append(r["source"])
        samples_flat.append(s)
scores_flat = score_cometkiwi(sources_flat, samples_flat)

best_scores = []
idx = 0
for r in rows:
    sent_scores = scores_flat[idx: idx + N]
    idx += N
    best_scores.append(max(sent_scores))

mean_best_of_n = sum(best_scores) / len(best_scores)

result = {
    "pair": PAIR, "model": MODEL_NAME, "n": N, "n_sentences": N_TEST,
    "mean_best_of_n_cometkiwi": mean_best_of_n,
}
out_scores_path = Path("results/phase3_bestofn_ende_scores.json")
with out_scores_path.open("w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
print(f"Wrote scores to {out_scores_path}")
