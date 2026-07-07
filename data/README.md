# Data

This directory is gitignored (except this file) — test sets are either downloaded
via script or copied from another project's outputs, not committed here.

## Layout

- `raw/` — test sets as obtained from their original source, unmodified.
- `processed/` — cleaned/filtered/tokenized versions derived from `raw/`, produced
  by scripts in `src/agentic_mt/` (document how each processed file was generated
  below as it's added).

## Sources

Document each dataset added to `raw/` here: name, source URL or paper, license,
language pairs covered, and the date it was pulled.

<!-- Example:
- `wmt24-en-de.tsv` — WMT24 General MT test set, en-de, downloaded 2026-07-06 from
  https://github.com/wmt-conference/wmt24-news-systems, CC-BY-SA.
-->
