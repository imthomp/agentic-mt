"""Shared multi-turn chat wrapper around a local open-source instruct model.

All four translation conditions use the SAME model and the SAME low-level
chat function — only the prompts (and whether prior turns are kept) differ.
This is what "same model" in the experiment brief means in practice.

Supports three model shapes, auto-detected:
- Decoder-only chat models with a real chat template (Llama, Qwen, etc.) —
  uses apply_chat_template.
- Encoder-decoder instruction models with no chat template, e.g. Aya-101
  (CohereForAI/aya-101 — chosen in Phase 3 because, unlike the newer
  Aya-23/Aya-Expanse models, it actually covers Hausa/Khmer/Pashto/Xhosa/Zulu).
- Decoder-only models with NO chat template either, e.g. BLOOMZ
  (bigscience/bloomz-7b1, covers xh/zu/sw but not ha) and InkubaLM
  (lelapa/InkubaLM-0.4B, covers ha/xh/zu/sw specifically but is tiny and
  needs trust_remote_code — a custom Llama subclass).

The latter two both get multi-turn context as a manually-formatted flat
transcript — functionally equivalent to a real chat template (the model
sees full prior context either way), just without special role tokens.
"""

import logging

logger = logging.getLogger(__name__)

_MODEL = None
_TOKENIZER = None
_MODEL_NAME = None
_IS_SEQ2SEQ = None
_HAS_CHAT_TEMPLATE = None
_MAX_POSITION_EMBEDDINGS = None

# Models whose custom architecture requires trust_remote_code=True to load.
TRUST_REMOTE_CODE_MODELS = {"lelapa/InkubaLM-0.4B"}

# Per-model generate() kwarg overrides.
GENERATE_KWARGS_OVERRIDES = {
    # InkubaLM's custom vulavulaslm.py forward pass was written against an
    # older transformers KV-cache API (its config declares
    # transformers_version 4.40.1; we run 4.57.6) and crashes indexing
    # past_key_values as the old tuple-of-tuples format — use_cache=False
    # routes around that incompatible code path entirely (recomputes the
    # full sequence each step — slower, but this model and our sequences
    # are both small enough that it doesn't matter). Also prone to
    # repetition loops at this size under pure greedy decoding — a
    # repetition penalty + no-repeat-ngram constraint fixes that.
    "lelapa/InkubaLM-0.4B": {"use_cache": False, "repetition_penalty": 1.3, "no_repeat_ngram_size": 3},
    # The "Answer:" completion cue (see PLAIN_INSTRUCTION_MODELS) fixed
    # BLOOMZ's immediate-EOS problem, but nudges it toward its QA-training
    # data's short-factoid-answer behavior for some prompts ("Jacques
    # Chirac" instead of a full translated sentence). min_new_tokens
    # forces it past that premature EOS without otherwise changing decoding.
    "bigscience/bloomz-7b1": {"min_new_tokens": 20},
}

# Models instruction-tuned on bare instruction->response pairs (BigScience's
# xP3 multitask format), NOT on role-tagged chat conversations. Wrapping
# their prompt in "System: ...\n\nUser: ...\n\nAssistant:" framing (as
# _flatten_messages does for Aya-101, which tolerates it fine) confused
# both BLOOMZ and InkubaLM badly — first full run showed 37-64% of
# l1_baseline outputs were verbatim echoes of the role-tag boilerplate
# ("I am a helpful assistant.", "Yes, I am.") instead of translations. For
# these models, drop role tags and system messages entirely; just
# concatenate turn contents as plain instruction text.
PLAIN_INSTRUCTION_MODELS = {"bigscience/bloomz-7b1", "lelapa/InkubaLM-0.4B"}


def _load(model_name: str):
    global _MODEL, _TOKENIZER, _MODEL_NAME, _IS_SEQ2SEQ, _HAS_CHAT_TEMPLATE, _MAX_POSITION_EMBEDDINGS
    if _MODEL is None or _MODEL_NAME != model_name:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info(f"Loading {model_name}")
        trust_remote_code = model_name in TRUST_REMOTE_CODE_MODELS
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        _IS_SEQ2SEQ = getattr(config, "is_encoder_decoder", False)
        _MAX_POSITION_EMBEDDINGS = getattr(config, "max_position_embeddings", None)
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        _HAS_CHAT_TEMPLATE = bool(getattr(_TOKENIZER, "chat_template", None))
        model_cls = AutoModelForSeq2SeqLM if _IS_SEQ2SEQ else AutoModelForCausalLM
        _MODEL = model_cls.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=trust_remote_code,
        )
        _MODEL_NAME = model_name
        logger.info(f"  is_seq2seq={_IS_SEQ2SEQ} has_chat_template={_HAS_CHAT_TEMPLATE} "
                    f"max_position_embeddings={_MAX_POSITION_EMBEDDINGS}")
    return _MODEL, _TOKENIZER, _IS_SEQ2SEQ, _HAS_CHAT_TEMPLATE, _MAX_POSITION_EMBEDDINGS


def unload() -> None:
    """Free the cached model from GPU memory (e.g. before loading COMET
    for scoring, to avoid both being resident at once on a 40GB GPU)."""
    global _MODEL, _TOKENIZER, _MODEL_NAME, _IS_SEQ2SEQ, _HAS_CHAT_TEMPLATE, _MAX_POSITION_EMBEDDINGS
    import gc
    _MODEL = None
    _TOKENIZER = None
    _MAX_POSITION_EMBEDDINGS = None
    _MODEL_NAME = None
    _IS_SEQ2SEQ = None
    _HAS_CHAT_TEMPLATE = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _flatten_messages(messages: list[dict], plain: bool = False) -> str:
    """Manually-formatted transcript for models with no chat template.

    `plain=True` drops system messages and role labels entirely (for
    models instruction-tuned without chat-role structure — see
    PLAIN_INSTRUCTION_MODELS); otherwise uses a role-tagged transcript
    (System:/User:/Assistant:), which Aya-101 handles fine."""
    if plain:
        contents = [m["content"] for m in messages if m["role"] != "system"]
        # A bare instruction with nothing after it triggers immediate EOS
        # for BLOOMZ (confirmed empirically); "Assistant:" instead makes it
        # echo the (now-dropped) system preamble. "Answer:" is a neutral
        # completion cue with neither failure mode, without hardcoding any
        # task-specific language name (this function doesn't know the
        # target language).
        return "\n\n".join(contents) + "\n\nAnswer:"

    lines = []
    for m in messages:
        role = m["role"].capitalize()
        lines.append(f"{role}: {m['content']}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


def _context_budget(tokenizer, max_new_tokens: int, max_position_embeddings: int | None = None) -> int:
    """Safe input-token truncation length: the model's real context window
    minus headroom for generation. tokenizer.model_max_length is often an
    unset huge sentinel value (not a real limit) — prefer the model
    config's max_position_embeddings when available (this is what actually
    caused InkubaLM, whose tokenizer reports no real limit but whose config
    caps at 2048, to silently exceed its context window)."""
    model_max = max_position_embeddings
    if not isinstance(model_max, int) or model_max <= 0:
        model_max = getattr(tokenizer, "model_max_length", 4096)
        if not isinstance(model_max, int) or model_max > 100_000:
            model_max = 4096
    return max(256, model_max - max_new_tokens)


def chat_turn(
    messages: list[dict],
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens: int = 512,
) -> str:
    """Run one more turn of a chat conversation and return the assistant reply.

    `messages` is the full conversation so far (system/user/assistant dicts),
    ending in the new user turn — the caller is responsible for building up
    conversation history across steps (see conditions/l1_cot.py).
    """
    model, tokenizer, is_seq2seq, has_chat_template, max_position_embeddings = _load(model_name)
    budget = _context_budget(tokenizer, max_new_tokens, max_position_embeddings)
    extra_kwargs = GENERATE_KWARGS_OVERRIDES.get(model_name, {})

    if is_seq2seq or not has_chat_template:
        plain = model_name in PLAIN_INSTRUCTION_MODELS
        prompt = _flatten_messages(messages, plain=plain)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=budget).to(model.device)
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id,
            **extra_kwargs,
        )
        if is_seq2seq:
            return tokenizer.decode(output[0], skip_special_tokens=True).strip()
        completion = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return completion.strip()

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=budget).to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        **extra_kwargs,
    )
    completion = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return completion.strip()


def sample_n(
    messages: list[dict],
    model_name: str,
    n: int,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
) -> list[str]:
    """Draw n independent samples for a single-turn prompt via temperature
    sampling -- for a best-of-N Level-1 baseline (proposal Sec. 7's "(b)
    inference-time scaled Level 1" condition), not used by any of the four
    L1 conditions, which are all greedy (do_sample=False in chat_turn)."""
    model, tokenizer, is_seq2seq, has_chat_template, max_position_embeddings = _load(model_name)
    budget = _context_budget(tokenizer, max_new_tokens, max_position_embeddings)
    extra_kwargs = GENERATE_KWARGS_OVERRIDES.get(model_name, {})

    if is_seq2seq or not has_chat_template:
        plain = model_name in PLAIN_INSTRUCTION_MODELS
        prompt = _flatten_messages(messages, plain=plain)
    else:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=budget).to(model.device)
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
        num_return_sequences=n, pad_token_id=tokenizer.eos_token_id, **extra_kwargs,
    )
    if is_seq2seq:
        return [tokenizer.decode(o, skip_special_tokens=True).strip() for o in output]
    prompt_len = inputs["input_ids"].shape[1]
    return [tokenizer.decode(o[prompt_len:], skip_special_tokens=True).strip() for o in output]
