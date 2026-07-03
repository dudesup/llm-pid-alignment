"""Held-out evaluation metrics (Section 9, 10): perplexity (capability forgetting) and
preference margin (constitutional/alignment retention)."""
from __future__ import annotations

import dataclasses
import math

import torch

from .tokenize_utils import encode_plain_text, encode_prompt_response, pad_batch


@dataclasses.dataclass
class HeldOutMetrics:
    perplexity: float
    preference_margin: float


@torch.no_grad()
def compute_perplexity(model, tokenizer, texts: list[str], device, max_len: int, batch_size: int = 4) -> float:
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        examples = [encode_plain_text(tokenizer, t, max_len) for t in batch_texts]
        input_ids, attention_mask, labels = pad_batch(examples, tokenizer.pad_token_id, device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :]
        target = labels[:, 1:]
        mask = target != -100
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        token_logprobs = torch.gather(logprobs, dim=-1, index=target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        total_nll += -(token_logprobs * mask).sum().item()
        total_tokens += mask.sum().item()
    mean_nll = total_nll / max(total_tokens, 1)
    return math.exp(mean_nll)


@torch.no_grad()
def _mean_response_logprob(model, tokenizer, prompt: str, response: str, device, max_len: int) -> float:
    ids, mask, labels = encode_prompt_response(tokenizer, prompt, response, max_len)
    input_ids, attention_mask, labels_t = pad_batch([(ids, mask, labels)], tokenizer.pad_token_id, device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    target = labels_t[:, 1:]
    resp_mask = target != -100
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    token_logprobs = torch.gather(logprobs, dim=-1, index=target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
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
    """mean(logP(chosen) - logP(rejected)), response tokens only, per-token averaged
    (Section 10, Figure 6) — per-token averaging avoids penalizing longer responses
    relative to length as a raw sum would.
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
) -> HeldOutMetrics:
    return HeldOutMetrics(
        perplexity=compute_perplexity(model, tokenizer, perplexity_texts, device, max_len),
        preference_margin=compute_preference_margin(
            model, tokenizer, preference_pairs, prompt_splitter, device, max_len
        ),
    )
