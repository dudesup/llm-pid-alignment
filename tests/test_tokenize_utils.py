"""Tests for pidlora/tokenize_utils.py's prompt/response truncation — no torch model
dependency, just a fake word-counting tokenizer to control token counts precisely.

Regression context: a Faza-0 smoke test (tiny random model, max_seq_len=64) hit
`loss: NaN` on ~40% of steps. Root cause: encode_prompt_response truncated the whole
(prompt + response) sequence from the right, so whenever the prompt alone exceeded
max_len — routine for hh-rlhf's multi-turn conversations — the response was dropped
entirely, leaving an example with zero supervised tokens. HF's internal cross-entropy
then divides 0/0 for that example. Same failure mode is reachable at the real
max_seq_len=512 with long enough conversations, just rarer — the smoke test's small
max_len just made it fall out immediately instead of showing up hours into a real run.
"""
from pidlora.tokenize_utils import IGNORE_INDEX, encode_prompt_response


class FakeTokenizer:
    """One token id per whitespace-separated word — lets tests control token counts
    exactly without needing a real HF tokenizer."""

    eos_token_id = 999

    def __call__(self, text: str, add_special_tokens: bool = False):
        words = text.split()
        return {"input_ids": list(range(1, len(words) + 1))}


def _n_word_text(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


class TestEncodePromptResponse:
    def test_response_survives_when_prompt_alone_exceeds_max_len(self):
        """The failure mode this fix targets: a 100-token prompt with a 5-token
        response and max_len=20 must not truncate the response away."""
        tok = FakeTokenizer()
        prompt = _n_word_text(100)
        response = _n_word_text(5)
        input_ids, attention_mask, labels = encode_prompt_response(tok, prompt, response, max_len=20)

        n_supervised = sum(1 for l in labels if l != IGNORE_INDEX)
        assert n_supervised == 6  # 5 response tokens + eos
        assert len(input_ids) <= 20
        assert len(labels) == len(input_ids) == len(attention_mask)

    def test_never_produces_zero_supervised_tokens_when_response_nonempty(self):
        """The actual invariant that prevents the NaN: as long as the response is
        non-empty, at least one label must survive truncation."""
        tok = FakeTokenizer()
        for prompt_len in [0, 1, 10, 50, 200, 1000]:
            for max_len in [4, 8, 16, 64]:
                _, _, labels = encode_prompt_response(
                    tok, _n_word_text(prompt_len), _n_word_text(2), max_len=max_len
                )
                n_supervised = sum(1 for l in labels if l != IGNORE_INDEX)
                assert n_supervised > 0, f"prompt_len={prompt_len} max_len={max_len} lost the whole response"

    def test_response_alone_exceeding_max_len_truncates_response_not_error(self):
        """Pathological case: response itself is longer than the whole budget. Prompt
        is dropped entirely and the response is truncated to max_len — no crash, no
        negative-length slicing."""
        tok = FakeTokenizer()
        input_ids, attention_mask, labels = encode_prompt_response(
            tok, _n_word_text(50), _n_word_text(30), max_len=10
        )
        assert len(input_ids) == 10
        assert all(l != IGNORE_INDEX for l in labels)  # entirely response tokens

    def test_no_truncation_needed_prompt_unchanged(self):
        tok = FakeTokenizer()
        input_ids, attention_mask, labels = encode_prompt_response(
            tok, _n_word_text(3), _n_word_text(2), max_len=100
        )
        assert len(input_ids) == 3 + 2 + 1  # prompt + response + eos
        assert labels[:3] == [IGNORE_INDEX] * 3
        assert labels[3:] != [IGNORE_INDEX] * 3

    def test_labels_and_attention_mask_shapes_match_input_ids(self):
        tok = FakeTokenizer()
        for prompt_len, response_len, max_len in [(0, 0, 10), (5, 5, 3), (20, 1, 15)]:
            input_ids, attention_mask, labels = encode_prompt_response(
                tok, _n_word_text(prompt_len), _n_word_text(response_len), max_len=max_len
            )
            assert len(input_ids) == len(attention_mask) == len(labels)
            assert all(a == 1 for a in attention_mask)
