"""Phase 6 extension: add Yoruba (89.7M ROOTS bytes) as an intermediate
point between en-sw (236M bytes, 50% clean) and en-xh/en-zu (14.3M/8.5M
bytes, 0% clean) -- per phase_6_summary.md's own "next steps," a point in
this gap is what would turn the four-point dose-response curve into an
actual regression instead of an eyeballed trend.
"""
import json
import logging
from pathlib import Path

from agentic_mt.pipelines.conditions import l1_baseline
from agentic_mt.pipelines.data import sample_test_and_tm_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "bigscience/bloomz-7b1"
test_df, _ = sample_test_and_tm_corpus("en-yo", 10, seed=42)

rows = []
for _, row in test_df.iterrows():
    result = l1_baseline.run(row["source"], "English", "Yoruba", MODEL_NAME, None)
    rows.append({"pair": "en-yo", "source": row["source"], "output": result["final_translation"]})
    logger.info("en-yo: done")

out_path = Path("results/phase_6_smoketest/smoketest_outputs_yo.jsonl")
with out_path.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"Wrote {len(rows)} rows to {out_path}")
