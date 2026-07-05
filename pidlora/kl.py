"""Forward KL against frozen top-k reference log-probs (design doc Section 5).

Materializing a full log-softmax over the vocab ([B, T, 152k] fp32, ~3.1 GB at a
control-set batch of 10 x ~500 tokens) alongside the 4-bit model + LoRA + optimizer
state risks OOM on a T4. Both reference-building and per-step KL evaluation instead:
  1. gather/topk directly on raw logits (top-k order is identical on logits and on
     log_softmax(logits), since log_softmax is just logits minus a per-position
     constant — the constant doesn't change relative ordering), and
  2. compute the normalizing logsumexp in chunks along the time dimension, so peak
     transient memory is bounded by [B, chunk_size, V] instead of [B, T, V].
The same truncated measurement is used for every branch, so comparisons between
branches are unaffected; only the absolute KL scale is biased low (tail handling,
Section 5).
"""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Optional

import torch

DEFAULT_LOGSUMEXP_CHUNK = 64


class EMAFilter:
    """KL_filt(t) = beta*KL_raw(t) + (1-beta)*KL_filt(t-1). Generic smoothing for
    monitoring/plotting — logged alongside the raw KL on every kl_eval so the curves
    aren't dominated by 50-prompt measurement noise. The record format is deliberately
    the one a future controller would consume, but nothing here depends on a controller
    existing."""

    def __init__(self, beta: float = 0.5):
        self.beta = beta
        self.value: Optional[float] = None

    def update(self, raw: float) -> float:
        if self.value is None:
            self.value = raw
        else:
            self.value = self.beta * raw + (1.0 - self.beta) * self.value
        return self.value

    def state_dict(self) -> dict:
        return {"beta": self.beta, "value": self.value}

    def load_state_dict(self, state: dict) -> None:
        self.beta = state["beta"]
        self.value = state["value"]


def chunked_logsumexp(logits: torch.Tensor, chunk_size: int = DEFAULT_LOGSUMEXP_CHUNK) -> torch.Tensor:
    """logits: [B, T, V] (any float dtype). Returns logsumexp over V as [B, T] fp32,
    computed chunk_size time-positions at a time so peak transient memory is
    [B, chunk_size, V] instead of [B, T, V]. Public: also used by evals.py, which needs
    the same normalizing constant to convert a gathered target logit into a log-prob."""
    b, t, _ = logits.shape
    out = torch.empty(b, t, device=logits.device, dtype=torch.float32)
    for start in range(0, t, chunk_size):
        end = min(start + chunk_size, t)
        out[:, start:end] = torch.logsumexp(logits[:, start:end, :].float(), dim=-1)
    return out


@dataclasses.dataclass
class ReferenceLogProbs:
    """Top-k log-probs of the frozen base model over the control set.

    topk_logprobs: [num_sequences, seq_len, k]  (fp16, CPU)
    topk_indices:  [num_sequences, seq_len, k]  (int32, CPU)
    attention_mask:[num_sequences, seq_len]     (bool, CPU) — True where a token
        contributes to the KL average (response tokens only, matching the SFT loss mask)
    fingerprint: sha256 of (model_name, k, control_texts) at build time — validated on
        load so a config change (control set, model, k) can't silently reuse a stale cache.
    """

    topk_logprobs: torch.Tensor
    topk_indices: torch.Tensor
    attention_mask: torch.Tensor
    fingerprint: Optional[str] = None

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "topk_logprobs": self.topk_logprobs,
                "topk_indices": self.topk_indices,
                "attention_mask": self.attention_mask,
                "fingerprint": self.fingerprint,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceLogProbs":
        blob = torch.load(path, map_location="cpu")
        blob.setdefault("fingerprint", None)
        return cls(**blob)

    def batch(self, indices: torch.Tensor) -> "ReferenceLogProbs":
        return ReferenceLogProbs(
            topk_logprobs=self.topk_logprobs[indices],
            topk_indices=self.topk_indices[indices],
            attention_mask=self.attention_mask[indices],
            fingerprint=self.fingerprint,
        )


def compute_fingerprint(control_texts: list[str], model_name: str, k: int, max_seq_len: int) -> str:
    """Includes max_seq_len: the reference is built from *tokenized, truncated*
    sequences, so a max_seq_len change alone (same control texts, same model) changes
    what actually gets measured — the raw-text hash alone would silently reuse a cache
    built against a different truncation and produce a subtly wrong KL, not an error."""
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(str(k).encode("utf-8"))
    h.update(str(max_seq_len).encode("utf-8"))
    for t in control_texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    return h.hexdigest()


@torch.no_grad()
def compute_reference_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    k: int = 1000,
    chunk_size: int = DEFAULT_LOGSUMEXP_CHUNK,
    fingerprint: Optional[str] = None,
) -> ReferenceLogProbs:
    """Run the (base, adapter-disabled) model over the control set and keep only the
    top-k log-probs per position. Call once before training; cache to disk (CPU/fp16).

    Never materializes a full [B, T, V] log-softmax tensor: top-k is taken on the raw
    logits (same order as on log-probs) and the normalizing constant is computed via
    chunked logsumexp, applied only to the small [B, T, k] gathered slice.
    """
    model.eval()
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, T, V], model dtype (bf16)
    # topk/gather stay on the raw (bf16) tensor — casting to fp32 here would materialize
    # a full [B, T, V] fp32 copy up front and make the chunking below pointless (the
    # whole point is to never hold more than [B, chunk_size, V] in fp32 at once).
    topk_raw, topk_indices = torch.topk(logits, k=k, dim=-1)  # [B, T, k], bf16
    log_z = chunked_logsumexp(logits, chunk_size=chunk_size)  # [B, T] fp32, casts only per-chunk
    topk_logprobs = topk_raw.float() - log_z.unsqueeze(-1)  # [B, T, k] fp32 — small, k << V

    return ReferenceLogProbs(
        topk_logprobs=topk_logprobs.to(dtype=torch.float16, device="cpu"),
        topk_indices=topk_indices.to(dtype=torch.int32, device="cpu"),
        attention_mask=response_mask.to(dtype=torch.bool, device="cpu"),
        fingerprint=fingerprint,
    )


def forward_kl_topk(
    current_logits: torch.Tensor, ref: ReferenceLogProbs, chunk_size: int = DEFAULT_LOGSUMEXP_CHUNK
) -> float:
    """Truncated forward KL(p_base || p_current), averaged per token then over sequences
    (Section 5 formal definition, truncated to the stored top-k support of p_base).

    current_logits: [B, T, V] raw logits from the current (adapter-active) model. Never
        materializes torch.log_softmax(current_logits) (a full [B, T, V] tensor) — gathers
        at the reference's top-k indices from the raw logits first, then normalizes with
        a chunked logsumexp.
    """
    device = current_logits.device
    ref_logprobs = ref.topk_logprobs.to(device=device, dtype=torch.float32)  # [B, T, k]
    ref_indices = ref.topk_indices.to(device=device, dtype=torch.long)  # [B, T, k]
    mask = ref.attention_mask.to(device=device)  # [B, T]

    # p_base renormalized over the stored top-k support (tail handling, Section 5)
    p_base = torch.softmax(ref_logprobs, dim=-1)  # [B, T, k]
    log_p_base = torch.log_softmax(ref_logprobs, dim=-1)

    # Gather at the reference's top-k indices from the RAW (model-dtype, typically bf16)
    # logits before any fp32 cast. Casting current_logits to fp32 here would materialize
    # a full [B, T, V] fp32 copy (~3 GB at a realistic control-set batch) — exactly the
    # transient the chunking below exists to avoid. Only the small gathered [B, T, k]
    # slice and the chunked logsumexp output need to be fp32.
    gathered_current = torch.gather(current_logits, dim=-1, index=ref_indices).float()  # [B, T, k]
    log_z_current = chunked_logsumexp(current_logits, chunk_size=chunk_size)  # [B, T] fp32, per-chunk casts only
    log_p_current_at_topk = gathered_current - log_z_current.unsqueeze(-1)  # [B, T, k]

    kl_per_token = (p_base * (log_p_base - log_p_current_at_topk)).sum(dim=-1)  # [B, T]

    mask_f = mask.float()
    denom_per_seq = mask_f.sum(dim=-1).clamp_min(1.0)
    kl_per_seq = (kl_per_token * mask_f).sum(dim=-1) / denom_per_seq  # [B]

    return kl_per_seq.mean().item()


def renormalization_floor(ref: ReferenceLogProbs) -> float:
    """Theoretical KL value when current == base exactly: -log(Z) per token, where Z is
    the fraction of base-model probability mass captured by the stored top-k support
    (see tests/test_kl.py::test_self_kl_floor_matches_minus_log_topk_mass for the
    derivation). Averaged the same way forward_kl_topk averages (per-token, masked to
    response tokens, then per-sequence) so it's directly comparable to a measured KL on
    a freshly-initialized adapter (B=0, so p_current ≡ p_base) — see
    train.py's pretrain sanity check.
    """
    z = ref.topk_logprobs.to(torch.float32).exp().sum(dim=-1)  # [B, T]
    floor_per_token = -z.log()
    mask_f = ref.attention_mask.float()
    denom = mask_f.sum(dim=-1).clamp_min(1.0)
    floor_per_seq = (floor_per_token * mask_f).sum(dim=-1) / denom
    return floor_per_seq.mean().item()
