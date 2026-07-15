"""Phase 3 addendum: best-of-N Level-1 baseline on en-ha.

The proposal (Sec. 7) specifies that evaluating agentic MT should compare
(b) inference-time-scaled Level 1 (best-of-N, MBR) against (c) full agentic
Level 2, calling this "particularly important": does the architecture add
value beyond just spending more compute? Phase 3's four conditions never
included (b) -- this fills that gap for en-ha, the low-resource pair, using
the same 100 sentences/seed as the original Phase 3 run.

N=5 temperature samples per sentence (same total generation budget as one
Phase 3 condition x5, well under the compute Phase 3 already spent across
its 4 conditions), scored with reference-free COMET-KIWI (consistent with
Phase 1's finding that this pair's QE is only weakly reliable -- selection
here is illustrative, not a strong ground-truth pick), best-of-5 taken per
sentence.
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
PAIR = "en-ha"
TGT_LANG = "Hausa"
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

out_path = Path("results/phase3_bestofn_enha.jsonl")
with out_path.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
logger.info(f"Wrote {len(rows)} rows to {out_path}")

# Score every sample (N per sentence) with reference-free COMET-KIWI, then
# take the best of each sentence's N -- this is the "best-of-N" score.
sources_flat, samples_flat = [], []
for r in rows:
    for s in r["samples"]:
        sources_flat.append(r["source"])
        samples_flat.append(s)

scores_flat = score_cometkiwi(sources_flat, samples_flat)

best_scores = []
per_sentence_all_scores = []
idx = 0
for r in rows:
    sent_scores = scores_flat[idx: idx + N]
    idx += N
    best_scores.append(max(sent_scores))
    per_sentence_all_scores.append(sent_scores)

mean_best_of_n = sum(best_scores) / len(best_scores)
mean_of_first_sample = sum(s[0] for s in per_sentence_all_scores) / len(per_sentence_all_scores)

result = {
    "pair": PAIR, "model": MODEL_NAME, "n": N, "n_sentences": N_TEST,
    "mean_best_of_n_cometkiwi": mean_best_of_n,
    "mean_single_sample_cometkiwi": mean_of_first_sample,
}
out_scores_path = Path("results/phase3_bestofn_enha_scores.json")
with out_scores_path.open("w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
print(f"Wrote scores to {out_scores_path}")
