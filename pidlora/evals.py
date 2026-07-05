"""Held-out evaluation metrics (Section 9, 10): perplexity (capability forgetting) and
preference margin (constitutional/alignment retention).

Same memory discipline as kl.py: never materialize a full [B, T, V] fp32 log-softmax.
Here we only need the log-prob at a single target index per position — gather that
index from the raw (model-dtype) logits first, then subtract the normalizing constant
from chunked_logsumexp, which is cheaper than the top-k case in kl.py since there's no
gather-of-k step at all, just gather-of-1.
"""
from __future__ import annotations

import dataclasses
import math

import torch

from .kl import DEFAULT_LOGSUMEXP_CHUNK, chunked_logsumexp
from .tokenize_utils import encode_plain_text, encode_prompt_response, pad_batch


@dataclasses.dataclass
class HeldOutMetrics:
    perplexity: float
    preference_margin: float


def _target_logprobs(logits: torch.Tensor, target: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """logits: [B, T, V] raw (model-dtype) logits. target: [B, T] token ids (any values
    where mask is False are irrelevant — clamp before calling). Returns log p(target) at
    each position as [B, T] fp32, without ever materializing log_softmax(logits)."""
    gathered = torch.gather(logits, dim=-1, index=target.unsqueeze(-1)).squeeze(-1).float()  # [B, T]
    log_z = chunked_logsumexp(logits, chunk_size=chunk_size)  # [B, T] fp32, per-chunk casts only
    return gathered - log_z


@torch.no_grad()
def compute_perplexity(
    model, tokenizer, texts: list[str], device, max_len: int, batch_size: int = 4,
    chunk_size: int = DEFAULT_LOGSUMEXP_CHUNK,
) -> float:
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        examples = [encode_plain_text(tokenizer, t, max_len) for t in batch_texts]
        input_ids, attention_mask, labels = pad_batch(examples, tokenizer.pad_token_id, device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :]  # raw model dtype, not cast here
        target = labels[:, 1:]
        mask = target != -100
        token_logprobs = _target_logprobs(logits, target.clamp_min(0), chunk_size)
        total_nll += -(token_logprobs * mask).sum().item()
        total_tokens += mask.sum().item()
    mean_nll = total_nll / max(total_tokens, 1)
    return math.exp(mean_nll)


@torch.no_grad()
def _mean_response_logprob(
    model, tokenizer, prompt: str, response: str, device, max_len: int,
    chunk_size: int = DEFAULT_LOGSUMEXP_CHUNK,
) -> float:
    ids, mask, labels = encode_prompt_response(tokenizer, prompt, response, max_len)
    input_ids, attention_mask, labels_t = pad_batch([(ids, mask, labels)], tokenizer.pad_token_id, device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]  # raw model dtype, not cast here
    target = labels_t[:, 1:]
    resp_mask = target != -100
    token_logprobs = _target_logprobs(logits, target.clamp_min(0), chunk_size)
    n_resp = resp_mask.sum().item()
    if n_resp == 0:
        return 0.0
    return (token_logprobs * resp_mask).sum().item() / n_resp


@torch.no_grad()
def compute_preference_margin(
    model,
    tokenizer,
    pairs: list[tuple[str, str]],
    prompt_splitter,
    device,
    max_len: int,
) -> float:
    """mean(logP(chosen) - logP(rejected)) — log-probs averaged per response token
    (length-normalized), response tokens only (Section 10, Figure 6). Per-token
    averaging avoids penalizing longer responses relative to length as a raw sum would.
    """
    margins = []
    for chosen, rejected in pairs:
        prompt_c, response_c = prompt_splitter(chosen)
        prompt_r, response_r = prompt_splitter(rejected)
        logp_chosen = _mean_response_logprob(model, tokenizer, prompt_c, response_c, device, max_len)
        logp_rejected = _mean_response_logprob(model, tokenizer, prompt_r, response_r, device, max_len)
        margins.append(logp_chosen - logp_rejected)
    return sum(margins) / max(len(margins), 1)


def evaluate_held_out(
    model,
    tokenizer,
    perplexity_texts: list[str],
    preference_pairs: list[tuple[str, str]],
    prompt_splitter,
    device,
    max_len: int,
    perplexity_batch_size: int = 4,
) -> HeldOutMetrics:
    return HeldOutMetrics(
        perplexity=compute_perplexity(model, tokenizer, perplexity_texts, device, max_len, batch_size=perplexity_batch_size),
        preference_margin=compute_preference_margin(
            model, tokenizer, preference_pairs, prompt_splitter, device, max_len
        ),
    )
