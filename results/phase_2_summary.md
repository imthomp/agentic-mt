# Phase 2: source-side triggering features for translation failure

## What this is

A modular feature-extraction + logistic-regression pipeline in
`src/agentic_mt/triggering/` testing whether cheap source-side signals
(length, OOV proxy, LLM-rated idiomaticity, parse depth, domain markers)
predict low translation quality, reusing Phase 1's low-resource WMT DA
rows (ha, km, ps, xh, zu — see `results/phase_1_summary.md`).

## Architecture (modular by design)

- `triggering/features/*.py` — one file per extractor, each exposing
  `extract(df) -> pd.Series`. Registered in `features/__init__.py`'s
  `FEATURE_EXTRACTORS` dict — swapping in a new extractor never touches
  `pipeline.py` or `run_experiment.py`.
- `triggering/data.py` — loads Phase 1's rows plus a `source_lang` column
  (Phase 1 collapses translation direction into one non-English `lang`
  code; feature extractors need to know which side is actually the source).
- `triggering/pipeline.py` — runs a configured list of extractors.
- `triggering/modeling.py` — per-language quantile-based low-quality
  labeling, standardized logistic regression, AUC, permutation importance,
  `joblib` model saving.
- `triggering/run_experiment.py` — CLI: `--lang X` runs full feature
  extraction for one language only (SLURM array-task mode); no `--lang`
  loads all per-language CSVs (if present) and runs labeling + modeling +
  reporting.

## Compute: this ran on SLURM, not the login node

Every feature is cheap **except** idiomaticity (LLM inference over ~39k
sentences with `meta-llama/Llama-3.1-8B-Instruct`, already cached locally
from the mqmbench project). Initial instinct was to subsample and run
interactively — corrected mid-session: this cluster has real GPU capacity
(`dwmatrix`, 8x A100), so the pipeline runs as **5 parallel per-language
SLURM jobs** (one GPU each, full row set, no subsampling) followed by a
CPU-only merge/modeling job gated on `--dependency=afterok`. See
`scripts/submit_triggering.sh` (mirrors
`multilingual-mqm-benchmark/scripts/slurm/submit_chain.sh`'s conventions,
including its mem=128G lesson: cgroup overhead on this cluster is much
larger than what login-node profiling suggests).

First submission: `ha` (23,983 rows) completed in 1h26m; `km`/`ps`/`xh`/`zu`
all failed with the same bug (below). Fixed and resubmitted just those 4 —
idiomaticity resumed from its JSONL cache (already fully scored before the
crash), so the rerun took under a minute each.

**Bug found and fixed**: `extract_domain_marker_count` crashed for every
language except `ha`. Only the `en-ha` direction has English-source rows
in this dataset (km/ps only have `X-en`; xh/zu is `xh-zu`/`zu-xh`, never
English at all) — for the other four languages the English-source mask is
entirely `False`, and `pandas`'s `.apply()` on an empty slice returns an
empty *string*-dtype result, which can't be assigned into a `float64`
column. Fixed in `triggering/features/domain.py` by guarding on
`en_mask.any()` before the assignment.

## Coverage caveats (expected, not bugs)

- `parse_depth` and `domain_marker_count` are English-source only (spaCy
  has no trained parser, and we have no term lexicon, for
  Hausa/Khmer/Pashto/Xhosa/Zulu) — so they're only non-null for the 10,812
  `en-ha` rows, the *only* English-source slice in this dataset.
- `sentence_length` is whitespace-token count; Khmer is written without
  spaces between words, so km's sentence_length values aren't comparable
  to the other languages'.
- `idiomaticity` is a small/mid-size open LLM's self-reported 0-1 rating —
  a cheap proxy, and a noisier one for these under-represented languages
  than it would be for English.

## Results (run `2026-07-07T00-03-37`)

Low-quality label = bottom 25% of `quality_score`, computed **per
language** (not a global cutoff — Phase 1 showed score distributions
differ substantially by language).

**Pooled univariate (all 5 languages together) vs. per-language
breakdown — these disagree, and the disagreement is the finding:**

| feature | pooled Pearson r | pooled AUC | per-language Pearson r range |
|---|---|---|---|
| oov_rate | **-0.33** | 0.513 | -0.045 to +0.021 |
| sentence_length | +0.19 | 0.503 | -0.035 to +0.026 |
| idiomaticity | -0.07 | 0.517 | -0.067 to +0.035 |

The pooled Pearson r for oov_rate (-0.33) looks like a real signal — but
per-language it's -0.02 to +0.02 for every one of the 5 languages, with
inconsistent sign. **The pooled correlation is almost entirely a
between-language artifact** (different languages have different baseline
OOV rates *and* different baseline quality scores), not a within-language
relationship — textbook Simpson's paradox. This is exactly why
`_univariate_table_by_lang` exists; report the per-language table, not the
pooled one, if this goes in a paper.

**Multivariate logistic regression models** (AUC, permutation importance):

| Model | Features | n | AUC |
|---|---|---|---|
| universal | sentence_length, oov_rate | 38,786 | 0.513 |
| with_idiomaticity | + idiomaticity | 38,786 | 0.520 |
| full_english_source | + parse_depth, domain_marker_count (en-ha only) | 10,812 | 0.564 |

## Interpretation

**None of these source-side features meaningfully predict translation
failure for these language pairs** — all AUCs are within ~0.01-0.06 of
chance (0.5). idiomaticity is the strongest single univariate/permutation
signal in every model it's in, but still weak. This is a negative result
worth reporting as-is: cheap, easily-computed source-side triggers
(length, OOV proxy, LLM idiomaticity rating, parse depth, keyword-based
domain markers) do not explain much of what makes a low-resource
translation fail, at least for ha/km/ps/xh/zu with these particular
feature implementations. Don't over-fit a narrative to `full_english_source`'s
higher 0.564 AUC — it's a single language-pair/direction slice (en-ha,
n=10,812), not independent confirmation across languages.

## Where things live

- Code: `src/agentic_mt/triggering/{data,pipeline,modeling,run_experiment}.py`,
  `src/agentic_mt/triggering/features/*.py`
- Config: `configs/triggering_low_resource.yaml` (production, SLURM),
  `configs/triggering_low_resource_smoketest.yaml` (tiny CPU sanity check)
- SLURM: `scripts/submit_triggering.sh`
- Per-language feature tables (stable, reused across merge runs):
  `results/triggering_low_resource/features/features_{lang}.csv`
- Latest merge/model run: `results/triggering_low_resource/2026-07-07T00-03-37/`
  (`report.md`, `feature_importance.png`, `model_*.joblib`,
  `univariate_correlations*.csv`)

## Next steps (not yet done)

- The weak AUCs suggest either (a) these particular feature
  implementations are too crude (whitespace tokenization, a keyword-list
  domain lexicon, a small-ish LLM's idiomaticity self-report), or (b)
  source-side features alone genuinely aren't where the triggering signal
  lives for these pairs — worth checking hypothesis (b) against a
  higher-resource pair (e.g. en-de, which has real MQM data via mqmbench)
  as a sanity check before concluding source-side triggering doesn't work
  here specifically.
- If pursuing triggering further, a real (not keyword-list) domain
  classifier and a proper frequency-list-based OOV measure (rather than
  the NLLB-tokenizer fragmentation proxy) would be the two highest-value
  upgrades — both slot in as drop-in replacements for the existing
  extractors without touching the pipeline.
- Saved `.joblib` models are per-run in the timestamped output dir; if
  reusing one as a triggering classifier later, note it was fit on
  Phase 1's WMT DA rows, not on any target application's actual traffic.
