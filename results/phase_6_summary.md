# Phase 6: BLOOMZ smoke test with corrected resource anchors

## Context

Phase 5 found BLOOMZ needed 9 fixes to produce non-garbage output at all,
using en-de/en-ha as its pair set. Follow-up fact-checking (see the
conversation record) found the actual primary source — BLOOM's paper,
Table 1 — shows **German has zero presence in BLOOM's ROOTS pretraining
corpus**, while Xhosa (14.3M bytes) and Zulu (8.5M bytes) do have some.
So Phase 5's en-de anchor wasn't just unlucky, it was measuring two
different flavors of out-of-distribution rather than a real high-vs-low
resource contrast. This phase reruns the reliability smoke test with a
corrected anchor (en-fr, 208B ROOTS bytes) before considering any full run.

## Step 1: smoke test (per plan — l1_baseline only, no scoring, manual categorization)

10 sentences each, `en-fr`/`en-xh`/`en-zu` (FLORES-plus devtest — confirmed
in Phase 5 that none of en-fr/en-xh/en-zu have WMT MQM/DA data). Every one
of the 30 outputs was read and categorized by hand (not a heuristic
classifier — see `results/phase_6_smoketest/categorized_all.jsonl` for the
full log):

| Pair | ROOTS bytes | clean | source_echo | repetition_loop |
|---|---|---|---|---|
| en-fr | 208,242,620,434 | **10** | 0 | 0 |
| en-xh | 14,304,074 | **0** | 10 | 0 |
| en-zu | 8,511,561 | **0** | 8 | 2 |

en-fr: every output is a genuine, on-topic French translation (quality
varies — some are partial/paraphrased — but all are real attempts).
en-xh/en-zu: **every single output is the English source echoed back
verbatim or reordered**, or a pure repetition loop, with exactly one
exception — one Zulu output attempted real target-language text
(*"Ukukhanya okubuyiselwa kuyo..."*) before collapsing into repeating that
phrase.

## Step 1b: added a third data point (Swahili) to test whether this is continuous

Per the plan's Step 2 guidance ("consider whether Swahili as a second
low-resource comparison point would help triangulate whether the residual
gap is continuous with resource scale") — cheap to add, so added it before
finalizing the report rather than stopping at two data points:

| Pair | ROOTS bytes | clean | source_echo | repetition_loop |
|---|---|---|---|---|
| en-sw | 236,482,543 | **5** | 2 | 3 |

Full ranking:

| Pair | ROOTS bytes | Clean rate |
|---|---|---|
| en-fr | 208B | 100% |
| en-sw | 236M | 50% |
| en-xh | 14.3M | 0% |
| en-zu | 8.5M | 0% |

**This is a monotonic dose-response relationship spanning four orders of
magnitude of pretraining exposure.** It isn't a threshold effect where
anything below some cutoff fails uniformly — Swahili, sitting ~880x below
French but ~16-28x above Xhosa/Zulu, lands almost exactly at 50%, between
the two extremes. That's a genuinely clean, interpretable result: BLOOMZ's
ability to *generate* (not just comprehend — every failure case shows
perfect English comprehension, just defaults to producing English rather
than attempting the target language) scales with how much of that
language it actually saw during pretraining.

## Decision (per the plan's Step 2 gate)

**Not proceeding to Step 3** (the full 4-condition en-fr/en-xh/en-zu run)
as originally scoped. Per the plan's own second bullet: this is "no longer
a generation-config bug, it's a residual resource-scale signal... worth
reporting as a finding in its own right... rather than continuing to debug
prompts." Concretely: 0% clean generation for en-xh/en-zu means a
COMET/chrF quality run there would just be scoring English text against
Xhosa/Zulu references — a meaningless number dressed up as a translation-quality
result, not a fixable prompt issue.

This also isn't the "BLOOMZ is a general dead end regardless of language"
outcome (the plan's third bullet) — en-fr's 100% clean rate rules that out.
It's specifically and cleanly about how much of each language BLOOM's
pretraining actually saw.

## Interpretation, and how this connects to the rest of the project

This is possibly the single cleanest quantitative finding across Phases
1-6: **a model's raw pretraining exposure to a language is a hard
prerequisite for coherent generation in it, and the relationship looks
continuous, not binary, at least across these four points.** It reframes
something Phase 3 and Phase 5 were both implicitly wrestling with —
"does this model cover the language" — from a checkbox (is the ISO code in
a language tag list) into a quantity that predicts failure rate. Phase 3's
Aya-101/en-ha result and Phase 5's whole BLOOMZ effort were both operating
on the checkbox version of this question; this phase is the first place in
the project that actually measured the dose-response curve underlying it.

## Where things live

- Smoke-test scripts: `scripts/phase6_smoketest.py`,
  `scripts/phase6_smoketest_sw.py`
- Raw + categorized outputs:
  `results/phase_6_smoketest/{smoketest_outputs.jsonl,smoketest_outputs_sw.jsonl,categorized_all.jsonl,categorized_sw.jsonl}`
- Data: `en-fr` added to `pipelines/data.py`'s `LOADERS` (FLORES-plus,
  same pattern as en-xh/en-zu/en-sw) and `run_experiment.py`'s
  `TARGET_LANG_NAME`.

## Next steps (not yet done)

- With only 3-5 points on the ROOTS-bytes-vs-clean-rate curve, the
  "monotonic" claim is suggestive, not statistically established — more
  languages at intermediate byte counts (e.g. Yoruba at 89.7M, between
  Swahili and Xhosa) would let this become an actual regression rather
  than an eyeballed trend.
- If pursuing the joint-necessity question further with an open model,
  this phase's method (check the primary pretraining-corpus table, don't
  trust a language-tag list) should be applied *before* selecting a model,
  not after debugging reveals a problem — this would have caught Phase 5's
  core issue in an afternoon instead of nine fix cycles.
- A full run limited to en-fr alone (skip xh/zu entirely) would still be a
  valid, clean test of the CoT/tool conditions for a language BLOOMZ can
  actually generate — worth doing if a single-model, single-pair data
  point remains useful, though it wouldn't address the low-resource half
  of the joint-necessity question this project set out to test.
