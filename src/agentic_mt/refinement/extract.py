"""Extract a clean translation from a (possibly verbose) model response.

Needed because Qwen2.5-32B-Instruct frequently prefaces its answer with
explanation ("Given the feedback... the revised German sentence would
be:") or echoes the prompt's "Target:" label rather than outputting the
translation alone — even with an explicit "output only the translation"
instruction, this is defensive backup. Caught via a real bug: without
this, refine-step outputs (mostly English explanation text) were being
scored as the "translation," producing a spurious ~0.18 COMET-KIWI
collapse from iteration 0 to iteration 1 in the first full run — not a
genuine refinement effect.
"""

import re

_LABEL_RE = re.compile(r"^\s*(target|translation)\s*:\s*", re.IGNORECASE)


def extract_translation(text: str) -> str:
    text = text.strip()

    # If a short preamble line ends in ":" (e.g. "...would be:"), the
    # translation is whatever comes after the LAST such line.
    lines = text.split("\n")
    colon_line_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.endswith(":") and len(stripped) < 200 and i < len(lines) - 1:
            colon_line_idx = i
    if colon_line_idx is not None:
        text = "\n".join(lines[colon_line_idx + 1:]).strip()

    text = _LABEL_RE.sub("", text).strip()

    # Drop any trailing explanation paragraph after the translation itself.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if paragraphs:
        text = paragraphs[0]

    return text.strip()
