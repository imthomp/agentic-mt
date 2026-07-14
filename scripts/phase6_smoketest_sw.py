"""Phase 6 extension: add Swahili (236M ROOTS bytes) as a triangulation
point between en-fr (208B bytes, 100% clean) and en-xh/en-zu (14.3M/8.5M
bytes, 0% clean) -- does the failure scale continuously with resource size,
or is xh/zu specifically below some threshold that sw clears?
"""
import json
import logging
from pathlib import Path

from agentic_mt.pipelines.conditions import l1_baseline
from agentic_mt.pipelines.data import sample_test_and_tm_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "bigscience/bloomz-7b1"
test_df, _ = sample_test_and_tm_corpus("en-sw", 10, seed=42)

rows = []
for _, row in test_df.iterrows():
    result = l1_baseline.run(row["source"], "English", "Swahili", MODEL_NAME, None)
    rows.append({"pair": "en-sw", "source": row["source"], "output": result["final_translation"]})
    logger.info("en-sw: done")

out_path = Path("results/phase_6_smoketest/smoketest_outputs_sw.jsonl")
with out_path.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"Wrote {len(rows)} rows to {out_path}")
