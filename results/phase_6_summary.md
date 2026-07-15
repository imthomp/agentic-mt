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

## Step 1c: added a fourth data point (Yoruba) to fill the gap between Swahili and Xhosa/Zulu

Per this file's own "Next steps" (below, written before this addition): only 3-5 points on
the ROOTS-bytes-vs-clean-rate curve made "monotonic" suggestive, not established, and an
intermediate-exposure language between Swahili (236M bytes) and Xhosa/Zulu (14.3M/8.5M
bytes) was the specific gap that would test it. Yoruba (89.7M ROOTS bytes, confirmed against
the primary source — BLOOM paper, Table 1) sits in exactly that gap. Ran the same
smoke-test methodology: 10 sentences, `bigscience/bloomz-7b1`, `l1_baseline` only, hand
read and categorized (`results/phase_6_smoketest/smoketest_outputs_yo.jsonl`):

| Pair | ROOTS bytes | clean | source_echo | repetition_loop/garbled |
|---|---|---|---|---|
| en-yo | 89,695,835 | **0** | 5 | 5 |

**This does not confirm the monotonic story — it complicates it.** Yoruba has ~6x more
ROOTS bytes than Xhosa and ~10x more than Zulu, but its clean rate (0%) matches those two,
not the ~30-40% a straight interpolation between Swahili's 50% and Xhosa/Zulu's 0% would
predict. Two of the ten outputs were real attempts at Yoruba before collapsing into a
repetition loop (closer to Zulu's one exceptional case than to Swahili's partial successes),
but none finished clean. Revised ranking:

| Pair | ROOTS bytes | Clean rate |
|---|---|---|
| en-fr | 208B | 100% |
| en-sw | 236M | 50% |
| en-yo | 89.7M | 0% |
| en-xh | 14.3M | 0% |
| en-zu | 8.5M | 0% |

This is still a monotonic *non-increasing* relationship (clean rate never goes up as bytes
go down), so the qualitative floor-effect claim survives. But the *smooth dose-response*
framing from Step 1b — "a genuinely clean, interpretable result... scales with how much of
that language it actually saw" — does not survive Yoruba unchanged: the curve looks more
like a sharp drop somewhere between 236M and 89.7M bytes than a gradual ramp, and 89.7M was
supposed to be the point that would distinguish those two shapes. Independent support for
Yoruba being a genuine hard case for BLOOM, not just noise: the BLOOM paper's own Flores-101
results call out Yoruba specifically ("results are very poor between Swahili and Yoruba...
under-represented in BLOOM's training data") and its multilingual probing results (Table 12)
list Yoruba among the lowest-scoring languages of the 17 probed. Cross-checked before citing,
per this project's standing discipline.

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
1-6, revised after Step 1c: **a model's raw pretraining exposure to a
language is a hard prerequisite for coherent generation in it, but the
relationship looks like a floor/threshold effect, not a smooth
dose-response curve** — Yoruba's 89.7M bytes (more than Xhosa/Zulu, well
below Swahili) still produced 0% clean output, matching the two
lowest-exposure languages rather than landing between them and Swahili.
The directional claim (less exposure never produces a higher clean rate)
still holds across all five points; the shape claim (gradual, continuous
scaling) does not, and Step 1b's language asserting a smooth scaling
relationship was overclaiming from four widely-spaced points. It still
reframes something Phase 3 and Phase 5 were both implicitly wrestling
with — "does this model cover the language" — from a checkbox (is the
ISO code in a language tag list) into a quantity that predicts failure
rate, just with a sharper, less continuous shape than first thought.

## Where things live

- Smoke-test scripts: `scripts/phase6_smoketest.py`,
  `scripts/phase6_smoketest_sw.py`, `scripts/phase6_smoketest_yo.py`
- Raw + categorized outputs:
  `results/phase_6_smoketest/{smoketest_outputs.jsonl,smoketest_outputs_sw.jsonl,smoketest_outputs_yo.jsonl,categorized_all.jsonl,categorized_sw.jsonl}`
- Data: `en-fr`/`en-yo` added to `pipelines/data.py`'s `LOADERS` (FLORES-plus,
  same pattern as en-xh/en-zu/en-sw) and `run_experiment.py`'s
  `TARGET_LANG_NAME`.

## Next steps (not yet done)

- Step 1c added the planned intermediate point (Yoruba) and it complicated
  the story rather than confirming it — the apparent drop now sits
  somewhere between Swahili's 236M and Yoruba's 89.7M bytes, a factor of
  ~2.6x. A language in that narrower gap (or several) would be needed to
  tell whether this is a sharp threshold or just a steeper-than-expected
  slope; five points still isn't enough to fit a real curve shape.
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
