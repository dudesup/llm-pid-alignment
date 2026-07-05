"""GPU-free tests for pidlora/evals.py's memory-safe log-prob computation. Both public
functions (compute_perplexity, _mean_response_logprob) route through
evals._target_logprobs — tested directly here with synthetic tensors, the same way
test_kl.py tests forward_kl_topk/compute_reference_logprobs directly rather than
through a full tokenizer+model pipeline.
"""
import math

import torch

from pidlora.evals import _target_logprobs


def pytest_approx(x, rel=1e-9):
    class _Approx:
        def __eq__(self, other):
            return math.isclose(other, x, rel_tol=rel)
    return _Approx()


def _dense_target_logprobs(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The old (pre-fix) formula: materializes the full log-softmax tensor, then
    gathers at the target index. Ground truth that the memory-safe version must match."""
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(logprobs, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)


class TestTargetLogprobs:
    def test_matches_dense_log_softmax_reference(self):
        """seq_len=100 > DEFAULT_LOGSUMEXP_CHUNK(64) so the chunking loop actually
        splits into more than one chunk."""
        vocab, seq_len, batch = 40, 100, 2
        torch.manual_seed(11)
        logits = torch.randn(batch, seq_len, vocab)
        target = torch.randint(0, vocab, (batch, seq_len))

        dense = _dense_target_logprobs(logits, target)
        chunked = _target_logprobs(logits, target, chunk_size=64)
        assert torch.allclose(dense, chunked, atol=1e-4)

    def test_chunk_size_does_not_change_the_result(self):
        vocab, seq_len, batch = 40, 100, 2
        torch.manual_seed(12)
        logits = torch.randn(batch, seq_len, vocab)
        target = torch.randint(0, vocab, (batch, seq_len))

        full = _target_logprobs(logits, target, chunk_size=seq_len)
        chunk_1 = _target_logprobs(logits, target, chunk_size=1)
        chunk_17 = _target_logprobs(logits, target, chunk_size=17)
        assert torch.allclose(full, chunk_1, atol=1e-4)
        assert torch.allclose(full, chunk_17, atol=1e-4)

    def test_never_materializes_full_size_float_tensor(self, monkeypatch):
        """Regression guard mirroring tests/test_kl.py's tripwire for the same anti-
        pattern (torch.log_softmax(logits.float(), dim=-1) on the full [B,T,V]),
        independently discovered in evals.py after the kl.py fix. Tracks both
        `.float()` and `.to(torch.float32)` casts."""
        calls: list[tuple[int, ...]] = []
        original_float = torch.Tensor.float
        original_to = torch.Tensor.to

        def recording_float(self):
            calls.append(tuple(self.shape))
            return original_float(self)

        def recording_to(self, *args, **kwargs):
            requests_fp32 = torch.float32 in args or kwargs.get("dtype") is torch.float32
            if requests_fp32 and self.dtype is not torch.float32:
                calls.append(tuple(self.shape))
            return original_to(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "float", recording_float)
        monkeypatch.setattr(torch.Tensor, "to", recording_to)

        vocab, seq_len, batch = 40, 130, 2  # seq_len > chunk_size=64
        torch.manual_seed(13)
        logits = torch.randn(batch, seq_len, vocab).to(torch.bfloat16)
        target = torch.randint(0, vocab, (batch, seq_len))

        _target_logprobs(logits, target, chunk_size=64)

        full_shape = (batch, seq_len, vocab)
        offending = [c for c in calls if c == full_shape]
        assert not offending, (
            f"an fp32 cast was called on the full [B,T,V] logits shape {full_shape} "
            f"({len(offending)}x) in evals._target_logprobs"
        )
