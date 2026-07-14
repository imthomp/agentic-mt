"""Phase 6, Step 1: BLOOMZ reliability smoke test with corrected anchors.

Samples 10 sentences each for en-fr (corrected high-resource anchor —
German has zero presence in BLOOM's ROOTS corpus; French has 208B bytes),
en-xh, en-zu. Runs l1_baseline only (no CoT chain — this checks basic
generation reliability, not the joint-necessity question). Logs raw
output for manual categorization; does NOT score with COMET/chrF (this is
a reliability gate, not a quality run — see results/phase_5_summary.md
for why: BLOOMZ needed 9 fixes before its output was even worth scoring).
"""

import json
import logging
from pathlib import Path

from agentic_mt.pipelines.conditions import l1_baseline
from agentic_mt.pipelines.data import sample_test_and_tm_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "bigscience/bloomz-7b1"
PAIRS = {"en-fr": "French", "en-xh": "Xhosa", "en-zu": "Zulu"}
N_TEST = 10
SEED = 42

out_dir = Path("results/phase_6_smoketest")
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
for pair, tgt_lang in PAIRS.items():
    test_df, _ = sample_test_and_tm_corpus(pair, N_TEST, SEED)
    for _, row in test_df.iterrows():
        result = l1_baseline.run(row["source"], "English", tgt_lang, MODEL_NAME, None)
        rows.append({"pair": pair, "source": row["source"], "output": result["final_translation"]})
        logger.info(f"{pair}: done")

out_path = out_dir / "smoketest_outputs.jsonl"
with out_path.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"Wrote {len(rows)} rows to {out_path}")
