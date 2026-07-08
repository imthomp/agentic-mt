# Phase 4: TEaR self-refinement vs. COMET-KIWI-guided refinement

## What this is

`src/agentic_mt/refinement/` implements Feng et al. (2025)'s TEaR
(Translate → Estimate → Refine) loop as condition A, and a variant
(condition B) that replaces the self-generated 3-shot MQM critique with
external COMET-KIWI scoring, run for 4 iterations on en-de.

## Scope decisions

- **en-de only**, not en-de + en-ha. Chosen deliberately to avoid Phase 3's
  model-coverage tension (a model that can hold a coherent multi-turn
  conversation vs. one that covers Hausa) — a 4-iteration loop is even more
  exposed to multi-turn brittleness than Phase 3's 4-step CoT chain was, so
  a clean replication check took priority this round.
- **Model: Qwen2.5-32B-Instruct** (already cached, genuinely
  conversation-tuned decoder LM) — specifically to avoid Aya-101's
  degenerate multi-turn behavior from Phase 3.
- **Same 100 en-de sentences as Phase 3** (same seed through
  `agentic_mt.pipelines.data.sample_test_and_tm_corpus`), for
  cross-experiment comparability.

## Architecture (modular by design)

- `refinement/prompts.py` — TEaR's Translate/Estimate/Refine templates
  transcribed from Feng et al. (2025) Appendix B (Tables 17-19, including
  the exact 3-shot MQM examples), plus condition B's refine prompt (adapted
  from the paper's own XCOMET-Score baseline, Table 22, substituting
  COMET-KIWI — the paper already tested "external scalar score instead of
  self-critique," just with XCOMET).
- `refinement/iterate.py` — shared outer loop; conditions differ only in
  their `step_fn` (freeze-or-refine decision), matching this project's
  running "swap the extractor, not the pipeline" pattern.
- `refinement/condition_a.py` — self-critique: calls the Estimate prompt,
  parses whether it found a real error (heuristic: the paper's own MQM
  format always includes `" - "` before an error span; an error-free
  segment has none), freezes if not, else refines.
- `refinement/condition_b.py` — external QE: scores the current translation
  with COMET-KIWI; freezes at/above `qe_threshold` (0.85), else refines
  using the scalar score as feedback.
- `refinement/extract.py` — see bug below.
- `refinement/analysis.py` — per-iteration means, iteration-of-peak +
  gain-from-iteration-1 (computed over iterations 1..N only, per the brief).

## Real bug found: first full run's numbers were garbage

First run (job 12611829) produced a **catastrophic** drop from iteration 0
(COMET-KIWI 0.71) to iteration 1 (0.53) for *both* conditions, then flat
decline — nothing like the paper's small effect sizes. Investigated by
reading the actual raw model output before trusting the numbers:

1. **Translate step**: the prompt template ends in `"Target:"` (verbatim
   from Table 17) to elicit completion; Qwen2.5-32B often echoed that label
   back (`"Target: Wir wissen, dass..."`), which got scored as part of the
   translation. Affected 26/200 iteration-0 outputs.
2. **Refine step (the real damage)**: refine responses were frequently
   verbose explanations, e.g. *"Given the feedback and aiming for a more
   accurate and natural-sounding translation, the revised German sentence
   would be:\n\nWir wissen, dass..."* — the **entire response**, mostly
   English commentary, was being scored as "the translation." This is a
   real behavioral difference from GPT-4o-mini (the paper's model), which
   apparently complies with "compose the final translation" more literally.

**Fix**: added an explicit "output only the translation, no explanation"
instruction to every template that produces a translation, plus a
defensive `extract_translation()` post-processor (strips a leading
`"Target:"/"Translation:"` label; if a short preamble line ends in `:`,
takes everything after it; drops any trailing paragraph after a blank
line). Validated against the actual 290 buggy refine responses from the
first run before re-submitting: 287/290 (99%) extracted cleanly, 0 empty
extractions, 3 residual edge cases still contained preamble (noted, not
fixed — diminishing returns past 99%). Re-ran; second run's iteration-0
score (0.810) matches the first run almost exactly, confirming the
translate/estimate logic itself was fine and the bug was purely in
output parsing.

**Lesson for future phases**: read a sample of raw model output before
trusting aggregate numbers, especially for any prompt whose literal
compliance depends on the model — don't assume a model will follow
"output only X" implicitly just because the paper's original model did.

## Results (run 2026-07-08, job 12692259)

| Condition | Iter 0 (IT) | Iter 1 | Iter 2 | Iter 3 | Iter 4 |
|---|---|---|---|---|---|
| A (TEaR self-critique), COMET-KIWI | 0.810 | 0.822 | **0.832** | 0.828 | 0.832 |
| B (COMET-KIWI-guided), COMET-KIWI | 0.810 | 0.809 | 0.818 | 0.809 | 0.816 |

Iteration-of-peak / gain (iterations 1..4 only, per the brief):

| Condition | Metric | Score@iter1 | Iteration of peak | Score@peak | Gain (iter1→peak) |
|---|---|---|---|---|---|
| A | cometkiwi | 0.822 | **4** | 0.832 | +0.011 |
| B | cometkiwi | 0.809 | **2** | 0.818 | +0.010 |
| A | comet | 0.845 | **4** | 0.852 | +0.007 |
| B | comet | 0.828 | **4** | 0.838 | +0.010 |
| A | chrf | 58.11 | **4** | 59.58 | +1.47 |
| B | chrf | 56.43 | **4** | 57.47 | +1.04 |

## Hypothesis result: not confirmed — and the paper doesn't actually predict it

The brief's hypothesis was "condition A should peak around iteration 1 and
plateau/decline (matching TEaR's reported pattern)." **This didn't
happen** — condition A peaks at iteration 4 (or 2, depending on metric),
not iteration 1, and shows no decline.

Worth flagging directly: re-reading Feng et al. (2025) §3.3 ("Iterative
Refinement Yields Additional Improvements") and Figure 4, **the paper's own
reported pattern is continuous, steady improvement through 8 iterations**,
not an early peak followed by decline — *"automatic metric scores improved
steadily with each iteration."* An early-peak-then-decline pattern does
appear in Phase 3's paper (Wu, Aycock & Monz 2025 — "Additional steps of
refinement... produce only marginal improvements, or occasional
degradation for Gemini"), which examined a different self-refinement
method (decomposition + repeated "translate again," no explicit
Estimate/freeze step). It's plausible the hypothesis as stated conflated
the two papers' findings. Our condition A result — continued, if modest
and non-monotonic, improvement through iteration 4 — is actually
**consistent with what TEaR's own paper reports**, just not with the
specific "peaks at 1" prediction in the brief.

## What the results do show

- **Freeze counts diverge sharply**: by iteration 4, condition A (self-critique)
  has frozen only 31/100 sentences — it keeps finding "errors" to fix in
  the other 69 every round. Condition B (COMET-KIWI ≥ 0.85) freezes 48/100
  by iteration 3 and plateaus there. External QE is a more decisive
  (or perhaps just cruder) stopping rule than self-critique.
- **Condition A's trajectory is smoother**; condition B's is a visible
  sawtooth (down at 1, up at 2, down at 3, up at 4 — see
  `quality_vs_iteration.png`). Plausible explanation: a single scalar
  COMET-KIWI score gives the LLM no error *location*, only "this is
  somewhat bad" — refining under that vague signal looks close to a
  coin-flip for sentences hovering near the threshold, whereas self-critique's
  span-level MQM feedback gives directed edits. This is a genuinely
  interesting secondary finding: **span-level feedback (even
  self-generated) may matter more than the source of the feedback**
  (self vs. external) for refinement stability — worth testing directly in
  a follow-up (e.g. condition B using XCOMET instead of COMET-KIWI, since
  XCOMET does provide error spans, matching Feng et al.'s own
  XCOMET-Span baseline).
- Neither condition shows the paper's clean +0.38-point-style gain at a
  single iteration; both are gaining roughly +0.01-0.02 by iteration 4 on
  COMET-KIWI. Plausibly Qwen2.5-32B's initial translations are already
  strong enough (IT COMET-KIWI 0.81, vs. GPT-4o-mini's ~0.80 in the paper)
  that there's less headroom for self-refinement to reclaim.

## Where things live

- Code: `src/agentic_mt/refinement/{prompts,iterate,condition_a,condition_b,extract,analysis,run_experiment}.py`
- Config: `configs/refinement_tear.yaml`
- SLURM: `scripts/submit_refinement.sh`
- Outputs: `results/refinement_tear/{outputs.csv,prompts.jsonl,iteration_means.csv,peak_analysis.csv,quality_vs_iteration.png,report.md}`

## Next steps (not yet done)

- Rerun condition B with XCOMET (span-level) instead of COMET-KIWI
  (scalar-only) to directly test the "span-level feedback matters more
  than feedback source" hypothesis raised above.
- Extend to en-ha once a model exists that's both conversation-tuned *and*
  covers Hausa (the same open question from Phase 3).
- The 3 residual un-parsed refine responses (of 290) suggest a small
  fraction of cases where Qwen ignores the "output only" instruction more
  thoroughly — worth a stricter regeneration-on-parse-failure retry if this
  experiment is extended to a larger sample.
