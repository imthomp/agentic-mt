"""Shared multi-turn chat wrapper around a local open-source instruct model.

All four translation conditions use the SAME model and the SAME low-level
chat function — only the prompts (and whether prior turns are kept) differ.
This is what "same model" in the experiment brief means in practice.

Supports both decoder-only chat models (Llama, Qwen, etc., via
apply_chat_template) and encoder-decoder instruction models like Aya-101
(CohereForAI/aya-101 — chosen here because, unlike the newer Aya-23/
Aya-Expanse models, it actually covers Hausa/Khmer/Pashto/Xhosa/Zulu).
Aya-101 has no chat template (single-turn text2text, not conversation-tuned),
so multi-turn context is given as a manually-formatted flat transcript
instead — functionally equivalent (the encoder sees full prior context
either way), just without special role tokens.
"""

import logging

logger = logging.getLogger(__name__)

_MODEL = None
_TOKENIZER = None
_MODEL_NAME = None
_IS_SEQ2SEQ = None


def _load(model_name: str):
    global _MODEL, _TOKENIZER, _MODEL_NAME, _IS_SEQ2SEQ
    if _MODEL is None or _MODEL_NAME != model_name:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info(f"Loading {model_name}")
        config = AutoConfig.from_pretrained(model_name)
        _IS_SEQ2SEQ = getattr(config, "is_encoder_decoder", False)
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        model_cls = AutoModelForSeq2SeqLM if _IS_SEQ2SEQ else AutoModelForCausalLM
        _MODEL = model_cls.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        _MODEL_NAME = model_name
    return _MODEL, _TOKENIZER, _IS_SEQ2SEQ


def unload() -> None:
    """Free the cached model from GPU memory (e.g. before loading COMET
    for scoring, to avoid both being resident at once on a 40GB GPU)."""
    global _MODEL, _TOKENIZER, _MODEL_NAME, _IS_SEQ2SEQ
    import gc
    _MODEL = None
    _TOKENIZER = None
    _MODEL_NAME = None
    _IS_SEQ2SEQ = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _flatten_messages(messages: list[dict]) -> str:
    """Manually-formatted transcript for models with no chat template."""
    lines = []
    for m in messages:
        role = m["role"].capitalize()
        lines.append(f"{role}: {m['content']}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


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
    model, tokenizer, is_seq2seq = _load(model_name)

    if is_seq2seq:
        prompt = _flatten_messages(messages)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(output[0], skip_special_tokens=True).strip()

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    completion = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return completion.strip()
