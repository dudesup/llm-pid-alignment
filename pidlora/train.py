"""CLI entrypoint tying together data / model / kl / evals / checkpoint / logging.
Everything that affects a result lives in this package — run_colab.ipynb only ever
calls `python -m pidlora.train --config <path> [--resume|--no-resume]`.

Scope note: static-alpha branches only (baseline, sweep). The PI controller and
threshold-heuristic branches are a later addition, not implemented here yet.

Usage:
    python -m pidlora.train --config config/baseline.yaml
    python -m pidlora.train --config config/sweep_a8.yaml --resume        # default
    python -m pidlora.train --config config/baseline.yaml --no-resume     # explicit restart
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

from . import checkpoint as ckpt
from . import data
from . import evals
from . import kl as kl_utils
from . import model as model_utils
from .config import RunConfig
from .logging_utils import MetricsLogger
from .tokenize_utils import encode_plain_text, encode_prompt_response, pad_batch, response_mask_from_labels

KL_PRETRAIN_CHECK_TOLERANCE = 0.02


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_control_tensors(tokenizer, control_texts: list[str], max_len: int):
    """Returns CPU tensors — callers move slices to the target device batch-wise
    (Section 5), so this never holds the full control set on GPU at once."""
    examples = [encode_plain_text(tokenizer, t, max_len) for t in control_texts]
    input_ids, attention_mask, labels = pad_batch(examples, tokenizer.pad_token_id, device="cpu")
    response_mask = response_mask_from_labels(labels)
    return input_ids, attention_mask, response_mask


def get_or_build_reference_logprobs(
    cfg: RunConfig, model, control_texts, control_input_ids, control_attention_mask, control_response_mask, device, logger
):
    cache_path = Path(cfg.reference_logprobs_cache)
    expected_fingerprint = kl_utils.compute_fingerprint(control_texts, cfg.model_name, cfg.topk_logprobs, cfg.max_seq_len)

    if cache_path.exists():
        cached = kl_utils.ReferenceLogProbs.load(cache_path)
        if cached.fingerprint == expected_fingerprint:
            return cached
        logger.log(
            step=0, event="reference_cache_stale",
            cached_fingerprint=cached.fingerprint, expected_fingerprint=expected_fingerprint,
        )

    chunks = []
    batch_size = cfg.kl_eval_batch_size
    with model_utils.adapter_disabled(model):
        for i in range(0, control_input_ids.shape[0], batch_size):
            ids = control_input_ids[i : i + batch_size].to(device)
            mask = control_attention_mask[i : i + batch_size].to(device)
            resp_mask = control_response_mask[i : i + batch_size]
            chunks.append(kl_utils.compute_reference_logprobs(model, ids, mask, resp_mask, k=cfg.topk_logprobs))

    ref = kl_utils.ReferenceLogProbs(
        topk_logprobs=torch.cat([c.topk_logprobs for c in chunks], dim=0),
        topk_indices=torch.cat([c.topk_indices for c in chunks], dim=0),
        attention_mask=torch.cat([c.attention_mask for c in chunks], dim=0),
        fingerprint=expected_fingerprint,
    )
    ref.save(cache_path)
    return ref


@torch.no_grad()
def compute_current_kl(model, control_input_ids, control_attention_mask, ref: kl_utils.ReferenceLogProbs, device, batch_size: int) -> float:
    model.eval()
    total_kl, total_n = 0.0, 0
    n = control_input_ids.shape[0]
    for i in range(0, n, batch_size):
        ids = control_input_ids[i : i + batch_size].to(device)
        mask = control_attention_mask[i : i + batch_size].to(device)
        ref_chunk = ref.batch(torch.arange(i, min(i + batch_size, n)))
        logits = model(input_ids=ids, attention_mask=mask).logits
        chunk_n = ids.shape[0]
        kl_val = kl_utils.forward_kl_topk(logits, ref_chunk)
        total_kl += kl_val * chunk_n
        total_n += chunk_n
    return total_kl / max(total_n, 1)


def train(cfg: RunConfig, resume: bool) -> None:
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(Path(cfg.output_dir) / "metrics.jsonl")
    logger.log(step=0, event="run_start", config=cfg.to_dict())

    model, tokenizer = model_utils.load_model_and_tokenizer(
        cfg.model_name, cfg.lora_r, cfg.alpha, cfg.lora_target_modules
    )
    model_utils.set_lora_scaling(model, alpha=cfg.alpha, r=cfg.lora_r)

    control_texts = data.build_control_set(cfg.control_set_size)
    held_out = data.build_holdout_set(cfg.holdout_wikitext_size, cfg.holdout_hhrlhf_size)

    # Disjointness is enforced by construction in data.py (shared shuffled permutation,
    # non-overlapping index ranges) — this is a cheap runtime guard against that
    # invariant silently breaking in a future edit. RuntimeError, not assert: a guard
    # whose entire purpose is catching a silent correctness violation must not be
    # removable by `python -O`.
    overlap = set(control_texts) & set(held_out.perplexity_texts)
    if overlap:
        raise RuntimeError(
            f"control/held-out overlap detected ({len(overlap)} texts) — Section 6 "
            "violated. This should be impossible by construction (data.py draws both "
            "from disjoint slices of a single shuffled wikitext permutation); if this "
            "fires, that invariant broke."
        )

    control_ids, control_mask, control_resp_mask = build_control_tensors(tokenizer, control_texts, cfg.max_seq_len)
    reference = get_or_build_reference_logprobs(
        cfg, model, control_texts, control_ids, control_mask, control_resp_mask, device, logger
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)

    start_step, extra_state = ckpt.maybe_resume(cfg.output_dir, model, optimizer, resume=resume)
    if start_step > 0:
        logger.log(step=start_step, event="resumed")

    kl_ema = kl_utils.EMAFilter(beta=cfg.kl_ema_beta)
    if extra_state.get("kl_ema") is not None:
        kl_ema.load_state_dict(extra_state["kl_ema"])

    if start_step == 0:
        # Sanity check before spending any compute: on a freshly-initialized adapter
        # (LoRA B=0, so p_current ≡ p_base exactly) the measured KL should equal the
        # renormalization floor. A mismatch means the KL pipeline (tokenization ->
        # masks -> top-k -> renormalization) is measuring something other than what it
        # claims to, and every subsequent minute of T4 time would be spent trusting a
        # broken metric — so this fails fast instead of just logging a warning. Only
        # meaningful at start_step==0: after resume the adapter is no longer identity.
        measured_kl = compute_current_kl(model, control_ids, control_mask, reference, device, cfg.kl_eval_batch_size)
        expected_floor = kl_utils.renormalization_floor(reference)
        diff = abs(measured_kl - expected_floor)
        ok = diff <= KL_PRETRAIN_CHECK_TOLERANCE
        logger.log(
            step=0, event="kl_pretrain_check", ok=ok,
            measured_kl=measured_kl, expected_floor=expected_floor, diff=diff,
        )
        if not ok:
            logger.close()
            raise RuntimeError(
                f"KL pretrain sanity check failed: measured={measured_kl:.4f} vs "
                f"expected floor={expected_floor:.4f} (diff={diff:.4f} > tolerance "
                f"{KL_PRETRAIN_CHECK_TOLERANCE}). The KL measurement pipeline is not "
                "producing the expected near-zero value for a fresh (B=0) adapter — "
                "stopping before burning further compute."
            )

    sft_stream = data.sft_examples(seed=cfg.seed)

    def next_microbatch(bs: int):
        examples = []
        for _ in range(bs):
            chosen = next(sft_stream)
            prompt, response = data.split_prompt_response(chosen)
            examples.append(encode_prompt_response(tokenizer, prompt, response, cfg.max_seq_len))
        return pad_batch(examples, tokenizer.pad_token_id, device)

    if start_step > 0:
        # sft_examples() owns a local RNG, not the global state checkpoint.py restores
        # (see data.py docstring) — fast-forward it past the examples already trained
        # on, or resume would silently replay the run's earliest batches.
        n_to_skip = start_step * cfg.grad_accum_steps * cfg.batch_size
        for _ in range(n_to_skip):
            next(sft_stream)
        logger.log(step=start_step, event="sft_stream_fast_forwarded", n_examples_skipped=n_to_skip)

    model.train()
    for step in range(start_step, cfg.total_steps):
        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            input_ids, attention_mask, labels = next_microbatch(cfg.batch_size)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss / cfg.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=cfg.grad_clip_max_norm)
        optimizer.step()

        logger.log(step=step, event="train_step", loss=accum_loss, grad_norm=float(grad_norm))

        do_periodic = cfg.is_full_logging
        is_last_step = step == cfg.total_steps - 1

        if do_periodic and (step % cfg.kl_eval_every == 0 or is_last_step):
            kl_raw = compute_current_kl(model, control_ids, control_mask, reference, device, cfg.kl_eval_batch_size)
            kl_filt = kl_ema.update(kl_raw)
            logger.log(step=step, event="kl_eval", kl_raw=kl_raw, kl_filt=kl_filt)
            model.train()

        if do_periodic and (step % cfg.holdout_eval_every == 0 or is_last_step):
            metrics = evals.evaluate_held_out(
                model, tokenizer, held_out.perplexity_texts, held_out.preference_pairs,
                data.split_prompt_response, device, cfg.max_seq_len,
                perplexity_batch_size=cfg.holdout_eval_batch_size,
            )
            logger.log(step=step, event="holdout_eval", perplexity=metrics.perplexity, preference_margin=metrics.preference_margin)
            model.train()

        if (step % cfg.checkpoint_every == 0 and step > start_step) or is_last_step:
            ckpt.save_checkpoint(cfg.output_dir, step + 1, model, optimizer, extra_state={"kl_ema": kl_ema.state_dict()})
            logger.log(step=step, event="checkpoint_saved")
            model.train()

    if not cfg.is_full_logging:
        # sweep branches: end-of-run metrics only (Section 15)
        kl_raw = compute_current_kl(model, control_ids, control_mask, reference, device, cfg.kl_eval_batch_size)
        metrics = evals.evaluate_held_out(
            model, tokenizer, held_out.perplexity_texts, held_out.preference_pairs,
            data.split_prompt_response, device, cfg.max_seq_len,
            perplexity_batch_size=cfg.holdout_eval_batch_size,
        )
        logger.log(
            step=cfg.total_steps - 1, event="final_eval", kl_raw=kl_raw,
            perplexity=metrics.perplexity, preference_margin=metrics.preference_margin,
        )

    logger.log(step=cfg.total_steps - 1, event="run_end")
    logger.close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    overrides = {}
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.max_steps is not None:
        overrides["total_steps"] = args.max_steps

    cfg = RunConfig.from_yaml(args.config, **overrides)
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main(sys.argv[1:])
