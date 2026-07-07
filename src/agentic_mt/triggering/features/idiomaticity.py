"""Idiomaticity/figurative-density feature: LLM-rated 0-1 score per sentence.

Cheap proxy, not a validated metric: we prompt an open-source instruct
model to self-report an idiomaticity rating. Caveat worth keeping in mind
when interpreting results — small-to-mid-size open LLMs have much weaker
coverage of Hausa/Khmer/Pashto/Xhosa/Zulu than of English, so ratings on
non-English source sentences are noisier and should be read as a weak
signal, not ground truth.

Resumable JSONL cache (idx -> rating), same pattern as mqmbench's
comet.py: safe to interrupt and re-run without losing completed ratings.
"""

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

_PROMPT_TEMPLATE = (
    "Rate how idiomatic or figurative the following sentence is, on a scale "
    "from 0.0 (fully literal, no idioms or figures of speech) to 1.0 "
    "(highly idiomatic/figurative). Respond with ONLY a number between 0.0 "
    "and 1.0, nothing else.\n\nSentence: {sentence}\n\nRating:"
)

_NUMBER_RE = re.compile(r"(\d*\.?\d+)")


def _parse_rating(text: str) -> float:
    match = _NUMBER_RE.search(text)
    if not match:
        return float("nan")
    value = float(match.group(1))
    return max(0.0, min(1.0, value))


def _read_cache(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for line in f:
            entry = json.loads(line)
            out[entry["idx"]] = entry["rating"]
    return out


def extract_idiomaticity(
    df: pd.DataFrame,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    sample_size: Optional[int] = None,
    seed: int = 42,
    cache_file: Optional[str] = None,
    batch_size: int = 8,
    device_map: str = "auto",
) -> pd.Series:
    """Rate idiomaticity for a (optionally subsampled) set of rows.

    Rows outside the sample get NaN — this is an expensive feature and is
    meant to be run on a bounded subset (see configs/*.yaml n_llm_sample),
    not silently skipped, so downstream code must treat NaN here as "not
    scored", not "definitely literal".
    """
    result = pd.Series(float("nan"), index=df.index)

    target_idx = df.index
    if sample_size is not None and sample_size < len(df):
        target_idx = df.sample(n=sample_size, random_state=seed).index

    cache_path = Path(cache_file) if cache_file else None
    cached = _read_cache(cache_path) if cache_path else {}
    for idx, rating in cached.items():
        if idx in result.index:
            result.loc[idx] = rating

    remaining_idx = [i for i in target_idx if i not in cached]
    if not remaining_idx:
        return result

    from transformers import pipeline
    generator = pipeline("text-generation", model=model_name, device_map=device_map)

    cache_handle = cache_path.open("a") if cache_path else None
    try:
        for start in range(0, len(remaining_idx), batch_size):
            batch_idx = remaining_idx[start : start + batch_size]
            prompts = [_PROMPT_TEMPLATE.format(sentence=df.loc[i, "source"]) for i in batch_idx]
            outputs = generator(
                prompts, max_new_tokens=8, do_sample=False,
                pad_token_id=generator.tokenizer.eos_token_id,
            )
            for i, prompt, out in zip(batch_idx, prompts, outputs):
                generated = out[0]["generated_text"]
                completion = generated[len(prompt):] if isinstance(generated, str) else ""
                rating = _parse_rating(completion)
                result.loc[i] = rating
                if cache_handle is not None:
                    cache_handle.write(json.dumps({"idx": int(i), "rating": rating}) + "\n")
            if cache_handle is not None:
                cache_handle.flush()
    finally:
        if cache_handle is not None:
            cache_handle.close()

    return result
