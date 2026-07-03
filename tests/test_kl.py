"""GPU-free tests for the truncated forward-KL math in pidlora/kl.py. Uses tiny
synthetic tensors (no model, no download) so these run on CPU in milliseconds."""
import math

import torch

from pidlora.kl import (
    EMAFilter,
    ReferenceLogProbs,
    compute_fingerprint,
    compute_reference_logprobs,
    forward_kl_topk,
    renormalization_floor,
)


def pytest_approx(x, rel=1e-9):
    class _Approx:
        def __eq__(self, other):
            return math.isclose(other, x, rel_tol=rel)
    return _Approx()


class _DenseModel(torch.nn.Module):
    """Stand-in for a HF model: returns fixed logits regardless of input, so
    compute_reference_logprobs can be exercised without a real model/tokenizer."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self._logits = logits

    def eval(self):
        return self

    def forward(self, input_ids=None, attention_mask=None):
        class _Out:
            pass
        out = _Out()
        out.logits = self._logits
        return out

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _dense_topk_logprobs(logits: torch.Tensor, k: int):
    """The old (pre-chunking) formula: materializes the full log-softmax tensor, then
    takes top-k. Used as the ground truth that the chunked implementation must match."""
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    return torch.topk(logprobs, k=k, dim=-1)


def _dense_forward_kl(current_logits: torch.Tensor, ref: ReferenceLogProbs) -> float:
    ref_logprobs = ref.topk_logprobs.float()
    ref_indices = ref.topk_indices.long()
    mask = ref.attention_mask

    p_base = torch.softmax(ref_logprobs, dim=-1)
    log_p_base = torch.log_softmax(ref_logprobs, dim=-1)

    current_logprobs = torch.log_softmax(current_logits.float(), dim=-1)  # full [B,T,V]
    log_p_current_at_topk = torch.gather(current_logprobs, dim=-1, index=ref_indices)

    kl_per_token = (p_base * (log_p_base - log_p_current_at_topk)).sum(dim=-1)
    mask_f = mask.float()
    denom = mask_f.sum(dim=-1).clamp_min(1.0)
    kl_per_seq = (kl_per_token * mask_f).sum(dim=-1) / denom
    return kl_per_seq.mean().item()


class TestChunkedMatchesDenseReference:
    """seq_len=100 > DEFAULT_LOGSUMEXP_CHUNK(64) so the chunking loop actually splits
    into more than one chunk — a seq_len <= 64 test would silently pass even if the
    chunk boundary logic were broken."""

    def test_matches_dense_log_softmax_reference(self):
        vocab, k, seq_len, batch = 40, 6, 100, 2
        torch.manual_seed(3)
        base_logits = torch.randn(batch, seq_len, vocab)
        current_logits = torch.randn(batch, seq_len, vocab)
        mask = torch.ones(batch, seq_len, dtype=torch.bool)

        dense_topk_logprobs, dense_topk_indices = _dense_topk_logprobs(base_logits, k)
        dense_ref = ReferenceLogProbs(
            topk_logprobs=dense_topk_logprobs.to(torch.float16),
            topk_indices=dense_topk_indices.to(torch.int32),
            attention_mask=mask,
        )

        model = _DenseModel(base_logits)
        chunked_ref = compute_reference_logprobs(
            model, input_ids=None, attention_mask=None, response_mask=mask, k=k, chunk_size=64
        )

        assert torch.equal(chunked_ref.topk_indices, dense_ref.topk_indices)
        assert torch.allclose(
            chunked_ref.topk_logprobs.float(), dense_ref.topk_logprobs.float(), atol=1e-3
        )

        dense_kl = _dense_forward_kl(current_logits, dense_ref)
        chunked_kl = forward_kl_topk(current_logits, chunked_ref, chunk_size=64)
        assert chunked_kl == pytest_approx(dense_kl, rel=1e-3)

    def test_chunk_size_does_not_change_the_result(self):
        """The chunk size is a memory/perf knob, not a semantic one."""
        vocab, k, seq_len, batch = 40, 6, 100, 2
        torch.manual_seed(4)
        base_logits = torch.randn(batch, seq_len, vocab)
        current_logits = torch.randn(batch, seq_len, vocab)
        mask = torch.ones(batch, seq_len, dtype=torch.bool)

        topk_logprobs, topk_indices = _dense_topk_logprobs(base_logits, k)
        ref = ReferenceLogProbs(
            topk_logprobs=topk_logprobs.to(torch.float16),
            topk_indices=topk_indices.to(torch.int32),
            attention_mask=mask,
        )

        kl_chunk_1 = forward_kl_topk(current_logits, ref, chunk_size=1)
        kl_chunk_17 = forward_kl_topk(current_logits, ref, chunk_size=17)
        kl_chunk_full = forward_kl_topk(current_logits, ref, chunk_size=seq_len)
        assert kl_chunk_1 == pytest_approx(kl_chunk_full, rel=1e-3)
        assert kl_chunk_17 == pytest_approx(kl_chunk_full, rel=1e-3)

    def test_never_materializes_full_size_float_tensor(self, monkeypatch):
        """Regression guard for the OOM fix itself, not just its output. A prior
        revision cast the whole [B, T, V] logits tensor to fp32 before gather/topk —
        semantically identical results (all the other tests here still passed), but the
        chunking became a no-op because the full-size fp32 copy already existed by the
        time it ran. Correctness tests can't catch that; this asserts directly that
        `.float()` is never called on a tensor shaped like the full logits, only on the
        small gathered/topk slices or chunk-sized slices."""
        calls: list[tuple[int, ...]] = []
        original_float = torch.Tensor.float

        def recording_float(self):
            calls.append(tuple(self.shape))
            return original_float(self)

        monkeypatch.setattr(torch.Tensor, "float", recording_float)

        vocab, k, seq_len, batch = 40, 6, 130, 2  # seq_len > chunk_size=64
        torch.manual_seed(7)
        base_logits = torch.randn(batch, seq_len, vocab).to(torch.bfloat16)
        current_logits = torch.randn(batch, seq_len, vocab).to(torch.bfloat16)
        mask = torch.ones(batch, seq_len, dtype=torch.bool)

        model = _DenseModel(base_logits)
        ref = compute_reference_logprobs(model, input_ids=None, attention_mask=None, response_mask=mask, k=k, chunk_size=64)
        forward_kl_topk(current_logits, ref, chunk_size=64)

        full_shape = (batch, seq_len, vocab)
        offending = [c for c in calls if c == full_shape]
        assert not offending, (
            f".float() was called on the full [B,T,V] logits shape {full_shape} "
            f"({len(offending)}x) — the OOM fix has regressed"
        )


class TestRenormalizationFloor:
    def test_matches_forward_kl_topk_self_kl(self):
        """renormalization_floor(ref) must equal forward_kl_topk(base_logits, ref) —
        it's the same quantity, computed directly from the reference instead of via a
        model forward pass (used for the pretrain sanity check in train.py, where
        running the model is the whole point of the comparison)."""
        vocab, k, seq_len = 20, 5, 3
        torch.manual_seed(5)
        base_logits = torch.randn(1, seq_len, vocab)
        topk_logprobs, topk_indices = _dense_topk_logprobs(base_logits, k)
        ref = ReferenceLogProbs(
            topk_logprobs=topk_logprobs.to(torch.float16),
            topk_indices=topk_indices.to(torch.int32),
            attention_mask=torch.ones(1, seq_len, dtype=torch.bool),
        )

        floor = renormalization_floor(ref)
        self_kl = forward_kl_topk(base_logits, ref)
        assert floor == pytest_approx(self_kl, rel=1e-3)

    def test_floor_respects_response_mask(self):
        vocab, k, seq_len = 20, 5, 4
        torch.manual_seed(6)
        base_logits = torch.randn(1, seq_len, vocab)
        topk_logprobs, topk_indices = _dense_topk_logprobs(base_logits, k)
        mask = torch.tensor([[True, True, False, False]])
        ref = ReferenceLogProbs(
            topk_logprobs=topk_logprobs.to(torch.float16),
            topk_indices=topk_indices.to(torch.int32),
            attention_mask=mask,
        )
        floor_masked = renormalization_floor(ref)

        ref_first_two = ReferenceLogProbs(
            topk_logprobs=topk_logprobs[:, :2].to(torch.float16),
            topk_indices=topk_indices[:, :2].to(torch.int32),
            attention_mask=torch.ones(1, 2, dtype=torch.bool),
        )
        floor_first_two = renormalization_floor(ref_first_two)
        assert floor_masked == pytest_approx(floor_first_two, rel=1e-3)


class TestEMAFilter:
    def test_first_update_returns_raw_value(self):
        ema = EMAFilter(beta=0.5)
        assert ema.update(1.0) == 1.0

    def test_matches_manual_recurrence(self):
        beta = 0.3
        ema = EMAFilter(beta=beta)
        raws = [1.0, 0.5, 0.8, 0.2]

        expected = raws[0]
        got = [ema.update(raws[0])]
        for r in raws[1:]:
            expected = beta * r + (1 - beta) * expected
            got.append(ema.update(r))

        assert got[-1] == pytest_approx(expected)

    def test_state_dict_roundtrip(self):
        ema = EMAFilter(beta=0.5)
        ema.update(1.0)
        ema.update(0.4)
        state = ema.state_dict()

        ema2 = EMAFilter(beta=0.9)
        ema2.load_state_dict(state)
        assert ema2.beta == 0.5
        assert ema2.value == ema.value


class TestComputeFingerprint:
    def test_deterministic(self):
        a = compute_fingerprint(["x", "y"], "model", 1000, 512)
        b = compute_fingerprint(["x", "y"], "model", 1000, 512)
        assert a == b

    def test_sensitive_to_max_seq_len(self):
        """The reference is built from tokenized, truncated sequences — a max_seq_len
        change alone (same texts, same model) changes what gets measured. Hashing only
        the raw text would let a stale cache be reused silently after that change."""
        a = compute_fingerprint(["x", "y"], "model", 1000, 512)
        b = compute_fingerprint(["x", "y"], "model", 1000, 1024)
        assert a != b

    def test_sensitive_to_model_and_k(self):
        base = compute_fingerprint(["x"], "model-a", 1000, 512)
        assert base != compute_fingerprint(["x"], "model-b", 1000, 512)
        assert base != compute_fingerprint(["x"], "model-a", 500, 512)

    def test_sensitive_to_text_content(self):
        a = compute_fingerprint(["hello", "world"], "model", 1000, 512)
        b = compute_fingerprint(["hello world"], "model", 1000, 512)  # concat ambiguity
        assert a != b


class TestForwardKLTopK:
    def test_self_kl_is_near_zero_when_topk_captures_nearly_all_mass(self):
        """Renormalizing p_base over the top-k support (Section 5 tail handling) means
        self-KL (current == base) is not *exactly* 0 in general: it has a small floor
        of -log(Z) where Z = sum of p_base over the stored top-k support. In the real
        deployment regime (k=1000 of a 152k vocab, tail mass "typically < 1-2%") Z is
        close to 1 so the floor is negligible. This test uses a deliberately peaked
        distribution so top-k captures ~100% of the mass, matching that regime — with
        a small/uniform vocab and few classes in top-k (as in a naive random test) Z
        can be much smaller and the floor becomes visibly nonzero; that is expected
        behavior of the renormalization, not a bug (see test below)."""
        vocab, k, seq_len = 50, 5, 3
        torch.manual_seed(0)
        # one dominant logit per position + noise on the rest -> top-k mass ~= 1
        base_logits = torch.full((1, seq_len, vocab), -10.0) + torch.randn(1, seq_len, vocab) * 0.1
        for t in range(seq_len):
            base_logits[0, t, t % vocab] = 20.0  # overwhelmingly dominant token
        base_logprobs = torch.log_softmax(base_logits, dim=-1)
        topk_logprobs, topk_indices = torch.topk(base_logprobs, k=k, dim=-1)

        ref = ReferenceLogProbs(
            topk_logprobs=topk_logprobs.to(torch.float16),
            topk_indices=topk_indices.to(torch.int32),
            attention_mask=torch.ones(1, seq_len, dtype=torch.bool),
        )

        kl = forward_kl_topk(base_logits, ref)
        assert abs(kl) < 1e-2

    def test_self_kl_floor_matches_minus_log_topk_mass(self):
        """Documents the renormalization floor exactly: for current == base, KL_computed
        should equal -log(Z) where Z is the base model's probability mass captured by
        the stored top-k support at each position, averaged the same way the KL itself
        is averaged. This is a property of the doc's chosen tail-handling scheme
        (Section 5), not an approximation error — asserting it here makes the behavior
        explicit instead of silently surprising whoever reads the KL curves later."""
        vocab, k, seq_len = 20, 5, 3
        torch.manual_seed(2)
        base_logits = torch.randn(1, seq_len, vocab)
        base_logprobs = torch.log_softmax(base_logits, dim=-1)
        topk_logprobs, topk_indices = torch.topk(base_logprobs, k=k, dim=-1)

        ref = ReferenceLogProbs(
            topk_logprobs=topk_logprobs.to(torch.float16),
            topk_indices=topk_indices.to(torch.int32),
            attention_mask=torch.ones(1, seq_len, dtype=torch.bool),
        )

        kl = forward_kl_topk(base_logits, ref)

        # Z per position from the *stored* (fp16-rounded) reference, matching what
        # forward_kl_topk itself sees
        z = topk_logprobs.to(torch.float16).to(torch.float32).exp().sum(dim=-1)  # [1, seq_len]
        expected_floor = (-z.log()).mean().item()
        assert kl == pytest_approx(expected_floor, rel=1e-2)

    def test_kl_is_nonnegative(self):
        torch.manual_seed(0)
        vocab, k, seq_len, batch = 30, 8, 4, 2
        base_logits = torch.randn(batch, seq_len, vocab)
        base_logprobs = torch.log_softmax(base_logits, dim=-1)
        topk_logprobs, topk_indices = torch.topk(base_logprobs, k=k, dim=-1)

        ref = ReferenceLogProbs(
            topk_logprobs=topk_logprobs.to(torch.float16),
            topk_indices=topk_indices.to(torch.int32),
            attention_mask=torch.ones(batch, seq_len, dtype=torch.bool),
        )

        current_logits = torch.randn(batch, seq_len, vocab)  # unrelated distribution
        kl = forward_kl_topk(current_logits, ref)
        assert kl >= -1e-4  # KL is nonnegative up to floating-point slack

    def test_masked_tokens_are_excluded(self):
        """Padding / prompt tokens (attention_mask=False) must not contribute to the
        averaged KL — Section 5: 'response tokens only'."""
        vocab, k, seq_len = 15, 4, 4
        torch.manual_seed(1)
        base_logits = torch.randn(1, seq_len, vocab)
        base_logprobs = torch.log_softmax(base_logits, dim=-1)
        topk_logprobs, topk_indices = torch.topk(base_logprobs, k=k, dim=-1)

        # only the first 2 of 4 positions are "real" (response) tokens
        mask = torch.tensor([[True, True, False, False]])
        ref = ReferenceLogProbs(
            topk_logprobs=topk_logprobs.to(torch.float16),
            topk_indices=topk_indices.to(torch.int32),
            attention_mask=mask,
        )

        current_logits = base_logits.clone()
        # corrupt only the masked-out positions — should have zero effect on the result
        current_logits[:, 2:, :] = torch.randn(1, 2, vocab) * 100
        kl_corrupted_but_masked = forward_kl_topk(current_logits, ref)

        current_logits_clean = base_logits.clone()
        kl_clean = forward_kl_topk(current_logits_clean, ref)

        assert kl_corrupted_but_masked == pytest_approx(kl_clean, rel=1e-3)
