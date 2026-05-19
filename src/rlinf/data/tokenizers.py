"""Minimal tokenizer helpers kept for the local embodied RLinf vendor."""

from transformers import AutoTokenizer


def hf_tokenizer(model_name_or_path: str, **kwargs):
    """Load a Hugging Face tokenizer with the defaults RLinf expects."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        **kwargs,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
