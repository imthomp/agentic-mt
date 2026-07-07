# agentic-mt

Agentic MT experiments: comparing translation conditions (L1, L1+CoT, L1+tool,
L1+CoT+tool) with quality estimation and source-side triggering features.

## Layout

- `data/` — test sets (gitignored; see [data/README.md](data/README.md) for sources).
- `src/agentic_mt/`
  - `qe/` — quality estimation wrappers (COMET, etc.)
  - `triggering/` — source-feature extraction (what triggers agentic behavior)
  - `pipelines/` — translation condition implementations (L1, L1+CoT, L1+tool, L1+CoT+tool)
  - `eval/` — correlation stats, BLEU/COMET/chrF scoring
- `notebooks/` — exploratory analysis, one notebook per experiment.
- `results/` — CSVs and plots, one timestamped subfolder per experiment run (gitignored).
- `configs/` — YAML configs per experiment; see [configs/schema.md](configs/schema.md)
  for the required fields.

## Setup

```bash
uv sync
```

## Running an experiment

```bash
uv run python -m agentic_mt.pipelines.<pipeline> --config configs/<name>.yaml
```
