# Phase 3: four translation conditions (L1_baseline, L1_cot, L1_tool, L1_cot_tool)

## What this is

`src/agentic_mt/pipelines/` implements and compares four translation
prompting strategies on the same 100 sampled sentences, same model, for a
high-resource pair (en-de, replication check against Wu, Aycock & Monz
2025) and a low-resource pair (en-ha, from Phase 1/2).

## Model choice: this took two tries

Started with `Llama-3.1-8B-Instruct` (already cached, low-risk default) —
jobs were running, then reconsidered given the low-resource focus of this
whole project. Checked actual language coverage of Cohere's Aya model
family via the HF API rather than assuming from marketing:

- **Aya-Expanse-32B / Aya-23-35B** (the newer, more-hyped Aya releases):
  cover only ~23 major languages (en/fr/de/es/.../vi) — **no Hausa at all**.
  Also gated (CC-BY-NC-4.0).
- **Aya-101** (`CohereForAI/aya-101`, the earlier, fully open Apache-2.0
  release): explicitly covers `hau`, `khm`, `pus`, `xho`, `zul` — literally
  all five of Phase 1/2's low-resource languages. Ungated, no download
  friction.

Cancelled the running Llama-3.1-8B jobs (~10 min in, no real cost) and
switched to Aya-101. This is the right call for a project centered on
low-resource translation — a model that structurally cannot represent the
target language is a worse confound than any prompting-strategy effect we're
trying to measure.

**Consequence**: Aya-101 is T5-based (encoder-decoder, `text2text-generation`,
not `conversational`-tagged) — it has no chat template and wasn't
multi-turn-tuned. `pipelines/llm.py` handles this by flattening
multi-turn context into a manually-formatted transcript for seq2seq models
(vs. `apply_chat_template` for decoder-only chat models like Llama), so the
same `chat_turn()` call works for either model family. See the important
caveat below — this workaround has real behavioral consequences.

## Architecture (modular by design)

- `pipelines/llm.py` — shared chat wrapper, dispatches on model architecture.
- `pipelines/data.py` — source/reference pairs for en-de (WMT MQM) and en-ha
  (WMT DA, Phase 1), each split into a 100-sentence test sample and a TM
  corpus (everything else, source-deduped so a test sentence's source can
  never leak into its own retrieval pool — see bug below).
- `pipelines/retrieval.py` — FAISS + LaBSE nearest-neighbor TM retrieval
  over the held-out corpus.
- `pipelines/conditions/prompts.py` — prompt templates transcribed as
  closely as possible from Wu, Aycock & Monz (2025) Appendix B (Figures
  9-12: research/decomposition → draft → refine → proofread, each a new
  turn with full prior context retained, per their Section 2). Their
  decomposition figure is cut off after "Idiomatic Expressions" in the
  PDF — replicated exactly what's visible rather than inventing the
  missing categories.
- `pipelines/conditions/{l1_baseline,l1_cot,l1_tool,l1_cot_tool}.py` — the
  four conditions; `l1_cot` and `l1_cot_tool` share the 4-step chain logic
  (`l1_cot.run_chain`), differing only in whether the decomposition prompt
  is augmented with a retrieved TM match and an explicit
  decide-whether-to-use-it instruction.
- `pipelines/scoring.py` — COMET-KIWI, reference-based COMET, chrF.
- `pipelines/analysis.py` — paired bootstrap significance between
  conditions, and the core hypothesis test (does the L1_cot_tool gap vs.
  L1_cot/L1_tool differ between en-de and en-ha? — unpaired
  difference-of-bootstrapped-gaps, since the two pairs are independent
  sentence samples).

## Bug found and fixed (before any GPU time spent)

`sample_test_and_tm_corpus`'s original source+reference anti-join let the
same *source* sentence leak into the TM corpus under a different
*reference* variant — en-de's MQM data has multiple post-edited references
per segment. Fixed by deduping to one row per source before splitting.
Caught by a simple overlap check before submitting any SLURM jobs.

## Compute: SLURM, not the login node

Two parallel per-pair jobs (`--pair en-de` / `--pair en-ha`, one GPU each,
100 sentences × 4 conditions, L1_cot/L1_cot_tool = 4 LLM calls each) +
a dependent CPU merge/analysis job. See `scripts/submit_pipelines.sh`
(mirrors `submit_triggering.sh`'s conventions). Aya-101 (49GB on disk,
~26GB in bf16) isn't cached, so pre-downloaded it on the login node with
`huggingface_hub.snapshot_download` before submitting with
`HF_HUB_OFFLINE=1`. Both pairs completed cleanly: en-de in 24 min, en-ha in
43 min (100 sentences × 10 generation calls each, plus COMET/COMET-KIWI/chrF
scoring). The LLM is explicitly unloaded from GPU memory (`llm.unload()`)
before COMET models load, to avoid both being resident at once.

## Important caveat: the CoT chain frequently degenerates with Aya-101

Spot-checked the full prompt/response logs
(`results/pipelines_l1_conditions/runs/prompts_{en-de,en-ha}.jsonl`) as
requested, specifically because L1_cot/L1_cot_tool showed 2-3x higher score
variance than L1_baseline/L1_tool. Found a real, systematic problem:

| | en-de l1_cot | en-de l1_cot_tool | en-ha l1_cot | en-ha l1_cot_tool |
|---|---|---|---|---|
| draft == refine == proofread (verbatim) | 80/100 | 78/100 | 72/100 | 86/100 |
| decomposition looks like a template echo | 90/100 | 43/100 | 6/100 | 26/100 |

For most sentences, Aya-101's "refinement" and "proofreading" steps
produce **byte-identical output** to the draft — no actual revision
happens, just deterministic (greedy) regurgitation. The decomposition step
itself is often degenerate too: for en-de it usually just echoes back
fragments of the prompt template itself (e.g. *"- Identify idiomatic
expressions that cannot be directly translated word-for-word into
German."*) rather than producing real analysis; one `l1_cot_tool`
decomposition response was literally the single word `"Yes"`.

**Why**: Aya-101 is a single-turn instruction-tuned seq2seq model, not
conversation-tuned (unlike the paper's GPT-4o-mini/Gemini-2.0-Flash). The
manually-flattened multi-turn transcript (see `llm.py`) gives it the right
*information*, but it wasn't trained to treat that transcript as a
conversation to continue coherently across turns the way a chat-tuned
decoder model would.

**What this means for interpreting the results below**: the "CoT hurts
translation quality" finding (which does directionally replicate the
paper) is real in the sense that it's what happened — but it's likely
driven substantially by Aya-101's degenerate multi-turn behavior rather
than a clean test of "does reasoning help translation" the way the
original paper tested it with genuinely conversation-capable models. Report
this replication as suggestive, not clean — and note it as a concrete
next step: rerun with a chat-tuned model that also covers Hausa (harder to
find — this is exactly the tension that drove the Aya-101 choice) before
treating the CoT finding as strong evidence either way.

L1_baseline and L1_tool (single-turn conditions) show no such degeneracy —
spot-checked outputs look like genuine, reasonable translations.

## Results (`results/pipelines_l1_conditions/report.md`, full run)

Mean scores per condition per pair:

| Pair | Condition | COMET-KIWI | COMET | chrF |
|---|---|---|---|---|
| en-de | l1_baseline | 0.804 | 0.821 | 53.3 |
| en-de | l1_cot | 0.766 | 0.782 | 50.4 |
| en-de | l1_tool | **0.819** | **0.843** | **59.5** |
| en-de | l1_cot_tool | 0.720 | 0.733 | 48.2 |
| en-ha | l1_baseline | 0.609 | 0.762 | 42.2 |
| en-ha | l1_cot | 0.563 | 0.698 | 35.9 |
| en-ha | l1_tool | **0.602** | **0.780** | **45.5** |
| en-ha | l1_cot_tool | 0.576 | 0.736 | 40.5 |

**L1_tool (mechanical TM injection, no reasoning) wins for both pairs on
every metric.** L1_cot and L1_cot_tool are significantly *worse* than
L1_baseline for en-de (paired bootstrap, p≤0.016 for COMET-KIWI/COMET); for
en-ha, L1_cot is significantly worse than baseline (p<0.001) but
L1_cot_tool is not significantly different from baseline (p=0.094 COMET,
p=0.026 COMET-KIWI — mixed).

**Core hypothesis test** — does the L1_cot_tool gap (vs. L1_cot, vs.
L1_tool) differ between en-de and en-ha? **Yes, significantly, across every
metric** (`resource_gap_comparison.csv`, p ranging 0.004-0.024):

- vs. L1_cot: adding the tool to the CoT chain makes things *worse* for
  en-de (gap -0.046 COMET-KIWI) but *better* for en-ha (gap +0.013 to
  +0.038 depending on metric) — the tool resource is relatively more
  useful, or at least less harmful, for the low-resource pair.
- vs. L1_tool: adding CoT reasoning on top of the tool makes things worse
  for both pairs, but *more* worse for en-de (-0.099 COMET-KIWI) than
  en-ha (-0.027).

This is a genuinely interesting, non-obvious pattern — but read it through
the degeneracy caveat above before treating it as a clean finding about
resource level per se, since Aya-101's multi-turn brittleness may itself
interact with resource level in ways this experiment can't cleanly separate
from a real reasoning effect.

## Where things live

- Code: `src/agentic_mt/pipelines/{llm,data,retrieval,scoring,analysis,run_experiment}.py`,
  `src/agentic_mt/pipelines/conditions/*.py`
- Config: `configs/pipelines_l1_conditions.yaml`
- SLURM: `scripts/submit_pipelines.sh`
- Per-pair outputs + full prompt/response logs (for spot-checking):
  `results/pipelines_l1_conditions/runs/{outputs,prompts}_{en-de,en-ha}.{csv,jsonl}`
- Analysis: `results/pipelines_l1_conditions/{condition_means,pairwise_significance,resource_gap_comparison}.csv`,
  `report.md`

## Next steps (not yet done)

- Rerun L1_cot/L1_cot_tool with a genuinely conversation-tuned model that
  also covers Hausa, to separate "CoT doesn't help" (the paper's finding)
  from "Aya-101 can't hold a multi-turn conversation" (what we may actually
  be measuring here). This is the single highest-priority follow-up.
- Consider an automated verbatim-repeat filter (flag rows where
  draft==refine==proofread) as a standard QC step for any future CoT-chain
  experiment, rather than relying on manual spot-checking to catch it.
- Extend to more low-resource pairs (km/ps/xh/zu) once the model-choice
  question above is resolved — right now en-ha is a single data point.
