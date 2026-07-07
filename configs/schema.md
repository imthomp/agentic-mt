# Experiment config schema

Every experiment config in `configs/` is a YAML file with the following required
fields:

| Field           | Type          | Description                                                                 |
|-----------------|---------------|-------------------------------------------------------------------------------|
| `language_pair` | string        | Source-target pair as `src-tgt` ISO 639-1 codes, e.g. `en-de`.                |
| `model_name`    | string        | Model identifier passed to the pipeline (e.g. `claude-sonnet-5`, `gpt-4o`).   |
| `sample_size`   | int           | Number of source segments to sample for the run.                             |
| `random_seed`   | int           | Seed for sampling and any stochastic pipeline steps, for reproducibility.     |
| `data_source`   | string        | Path (relative to `data/`) or identifier of the test set used.                |
| `output_dir`    | string        | Path (relative to `results/`) where this run's CSVs/plots are written.       |

## Conventions

- Config filenames should describe the run, e.g. `en-de_l1-cot_sonnet5.yaml`.
- `output_dir` should include a timestamp to keep runs from colliding, e.g.
  `results/en-de_l1-cot_sonnet5/2026-07-06T18-40/`.
- Additional fields specific to a pipeline (e.g. `condition: l1+cot+tool`,
  `temperature`, `max_tokens`) may be added as needed but the six fields above
  are required for every config so that `src/agentic_mt/eval` can load results
  across experiments uniformly.

## Example

```yaml
language_pair: en-de
model_name: claude-sonnet-5
sample_size: 200
random_seed: 42
data_source: raw/wmt24-en-de.tsv
output_dir: en-de_l1-cot_sonnet5
condition: l1+cot
```
