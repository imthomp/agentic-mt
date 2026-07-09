# Phase 5: attempting to unconfound Phase 3, and why it didn't work

## The goal

Phase 3's joint-necessity test (L1_baseline/L1_cot/L1_tool/L1_cot_tool) used
Aya-101 for both en-de and en-ha. Aya-101 isn't conversation-tuned (T5
seq2seq, single-turn), and its 4-step CoT chain visibly degenerated
(70-90% verbatim draft/refine/proofread repeats — see
`results/phase_3_summary.md`). Phase 5's goal was to rerun with open,
conversation-tuned models that also cover a low-resource language, to
separate "CoT doesn't help translation" from "this specific model can't
hold a multi-turn conversation."

Checked two candidates against actual language coverage (not marketing):

- **BLOOMZ-7b1** (BigScience, ungated, ordinary causal decoder, instruction-tuned
  via xP3): covers Xhosa/Zulu/Swahili, not Hausa.
- **InkubaLM-0.4B** (Lelapa AI, gated with manual approval, custom Llama
  subclass): covers Hausa/Xhosa/Zulu/Swahili specifically, but is only 0.4B
  parameters.

## Conclusion: neither model is usable for this experiment, and that's the finding

**InkubaLM** is a genuine capability floor, not a fixable prompt issue.
Even after fixing every mechanical bug (see below), its outputs were
incoherent and not even valid English, let alone a translation: *"Please
code the following dis from English or Hausa. probo only one english and
do su output anythin..."* No prompt engineering fixes an undertrained
0.4B model's inability to follow instructions. Confirmed via a targeted
smoke test, not assumed — dropped after that confirmation.

**BLOOMZ** took nine separate mechanical fixes to even reach a fair test of
its actual translation ability, and once it got there, the verdict was:
unreliable, not incapable. Working through them in order is itself the
interesting record of what "conversation-tuned + covers the language"
actually requires beyond checking a language list:

1. Model never pre-downloaded before an offline (`HF_HUB_OFFLINE=1`) SLURM
   job → immediate connection error.
2. InkubaLM's custom `vulavulaslm.py` forward pass, written against an
   older transformers KV-cache API (its config declares
   `transformers_version: 4.40.1`; we run 4.57.6), crashed indexing
   `past_key_values` as the old tuple format → fixed with `use_cache=False`.
3. InkubaLM gated repo only partially cached (one file fetched during an
   earlier access check, not the full model) → same offline-mode failure
   as #1, fixed with a full `snapshot_download`.
4. `tokenizer.model_max_length` reports an unset sentinel value for both
   models; my truncation fallback (4096) silently exceeded InkubaLM's real
   2048-token limit → fixed by reading `config.max_position_embeddings`.
5. **The big one**: wrapping the prompt in role-tagged
   `"System: ...\n\nUser: ...\n\nAssistant:"` framing (fine for Aya-101)
   confused both models badly — 37-64% of BLOOMZ's `l1_baseline` outputs
   were verbatim echoes of the *role-tag boilerplate itself* ("I am a
   helpful assistant.", "Yes, I am.") instead of translations, and
   InkubaLM fell into open-ended repetition loops ("we are not only in the
   same language, but also in the same language..."). Neither model was
   trained on chat-role structure (BLOOMZ: bare instruction→response pairs
   via xP3; InkubaLM: too small to generalize the framing at all). Fixed
   by dropping role tags/system messages entirely for these two models.
6. Removing the role tags also removed the "Assistant:" completion cue —
   with nothing to prompt continuation, BLOOMZ emitted EOS immediately,
   producing empty strings for 100% of outputs.
7. Guessed the cause was a repetition penalty I'd added defensively;
   removed it — still empty. Wrong diagnosis, correctly reverted.
8. Added a neutral `"Answer:"` cue back → BLOOMZ started producing real
   text again, but frequently truncated to 1-3 words ("Jacques Chirac",
   "2 ½", "British Garden Centres") — the cue nudged it toward xP3's
   QA-style short-factoid-answer training data instead of full-sentence
   generation.
9. Added `min_new_tokens=20` to force past that premature EOS — this
   finally produced full-length output. But the content was now
   **inconsistently correct**: some genuine, accurate German/Xhosa
   translations, alongside a large fraction that just echoed the English
   source verbatim unchanged, or fell into repetition loops ("2006. 2006.
   2006...", "Die Anstalt reported. Die Anstalt reported...").

Step 9 is where I stopped, per a hard-stop commitment made mid-debugging:
try one more targeted fix, and whatever the outcome, don't keep iterating
indefinitely. The result at that point was a real, not an artifactual,
reliability problem — BLOOMZ sometimes translates correctly and sometimes
doesn't even attempt to, with no clear prompt-level lever left to pull.
That's a legitimate stopping point, not a shortcut.

## What NOT to use from this phase

`results/pipelines_l1_conditions_bloomz/` and
`results/pipelines_l1_conditions_inkuba/` contain a full 4-pair /
2-pair run each, generated **before** fix #5 (the role-tag fix) landed.
Those numbers looked like an interesting cross-model contradiction of
Phase 3's finding (CoT helping for low-resource pairs, hurting for
en-de) when I first read them — but given steps 5-9 above, that pattern
is very likely just a garbage-rate artifact between conditions, not a
genuine translation-quality signal. **Do not cite those tables.** They're
left in place only as a record of what the failure looked like at that
stage; `results/phase_5_summary.md` (this file) is the source of truth for
what happened in Phase 5.

## Where this leaves Phase 3's joint-necessity question

Still open. Aya-101 remains the only model tested so far that's both
open and covers Hausa, and its confound (not conversation-tuned) is
well-documented but unresolved. This phase ruled out two plausible-looking
alternatives with concrete evidence, which narrows the search space even
though it didn't produce a positive result.

## Next steps (not yet done)

- A genuinely capable (7B+, modern, e.g. 2024+) open model that's
  *actually* conversation-tuned AND covers an African language would
  resolve this cleanly — worth checking newer releases' language cards
  directly (as this phase did for Aya) rather than assuming from
  reputation, given Aya-Expanse's Hausa gap and BLOOMZ's reliability gap
  were both invisible from surface-level marketing.
- If no such model turns up, the more defensible path may be accepting
  Aya-101 with its confound clearly flagged (as Phase 3 already does)
  rather than continuing to chase a clean unconfounded low-resource
  comparison — diminishing returns are real, as this phase demonstrates.
- The specific `min_new_tokens`/cue-engineering pattern developed here
  (`agentic_mt/pipelines/llm.py`'s `PLAIN_INSTRUCTION_MODELS` /
  `GENERATE_KWARGS_OVERRIDES`) is reusable infrastructure even though the
  two models tested through it didn't pan out — future models needing the
  same non-chat-template treatment can register there without new plumbing.
