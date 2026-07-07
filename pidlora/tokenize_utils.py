"""Shared prompt/response tokenization with prompt-masked labels (loss on response
tokens only — Section 5: "Loss computed on response tokens only (prompt is masked)").
Used by training, KL reference computation, and preference-margin evaluation so the
masking convention is identical everywhere it matters.
"""
from __future__ import annotations

import torch

IGNORE_INDEX = -100


def encode_prompt_response(tokenizer, prompt: str, response: str, max_len: int):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        response_ids = response_ids + [tokenizer.eos_token_id]

    # Truncate the PROMPT from the left (keep its tail), not the sequence from the
    # right — hh-rlhf's multi-turn prompts routinely exceed max_len on their own, and
    # right-truncation would silently drop the entire response, leaving an example
    # with zero supervised tokens. HF's internal cross-entropy then divides 0/0 for
    # that example (NaN loss), observed directly in a Faza 0 smoke test at
    # max_seq_len=64; the same failure mode is reachable at max_seq_len=512 with long
    # enough real conversations, just rarer. response_ids is also capped to max_len as
    # a last resort for the (pathological) case where the response alone exceeds it.
    response_ids = response_ids[:max_len]
    max_prompt_len = max(0, max_len - len(response_ids))
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:] if max_prompt_len > 0 else []

    input_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask, labels


def encode_plain_text(tokenizer, text: str, max_len: int):
    """Whole-sequence supervision (no prompt mask) — used for held-out perplexity on
    general-domain text (wikitext) where there is no prompt/response split."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max_len]
    if tokenizer.eos_token_id is not None and len(ids) < max_len:
        ids = ids + [tokenizer.eos_token_id]
    labels = list(ids)
    attention_mask = [1] * len(ids)
    return ids, attention_mask, labels


def pad_batch(examples: list[tuple[list[int], list[int], list[int]]], pad_id: int, device):
    """examples: list of (input_ids, attention_mask, labels). Right-pads to the batch max."""
    max_len = max(len(ex[0]) for ex in examples)
    input_ids, attention_mask, labels = [], [], []
    for ids, mask, lab in examples:
        pad_n = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_n)
        attention_mask.append(mask + [0] * pad_n)
        labels.append(lab + [IGNORE_INDEX] * pad_n)
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def response_mask_from_labels(labels: torch.Tensor) -> torch.Tensor:
    """Boolean mask of positions that carry real supervision (used as the KL
    "response tokens only" mask, Section 5)."""
    return labels != IGNORE_INDEX
