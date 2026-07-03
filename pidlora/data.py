"""Dataset construction (design doc Section 5, 6).

Three disjoint pools, deliberately kept apart:
  - control set   : general-domain, NOT hh-rlhf, NOT multiple-choice -> KL error signal
  - held-out set   : wikitext-2 (perplexity) + held-out hh-rlhf pairs (perplexity + preference
                     margin) -> final reporting only, NEVER seen by a controller
  - SFT stream     : hh-rlhf `chosen` responses -> what the model is actually trained on

Control/held-out composition is frozen independent of the run seed (Section 6: "frozen
before the experiment begins"). If it varied with cfg.seed, a second training seed
(Section 10's recommendation for a stronger claim) would measure KL against a different
control set, and the setpoint derived from one run would stop meaning anything for the
other. cfg.seed governs only training randomness (init, minibatch order) via
sft_examples(); control/held-out draws use fixed constants below.

Disjointness between the control and held-out wikitext slices is by construction: both
are index ranges into the *same* cached shuffled permutation (_shuffled_wiki), not two
independently-shuffled orderings of the same list. Two different shuffles offset-sliced
against each other give no disjointness guarantee at all — they're unrelated
permutations, so a "later" slice of one can freely contain items from an "earlier"
slice of the other.
"""
from __future__ import annotations

import dataclasses
import functools
import random
from typing import Iterator

from datasets import load_dataset

# Small curated set of factual-question / code-snippet prompts to diversify the control
# set beyond wikitext continuations (Section 5: "wikitext-2 sentences, factual questions,
# coding snippets"). Deliberately short and domain-general — not hh-rlhf, not MC.
_FACTUAL_AND_CODE_PROMPTS: tuple[str, ...] = (
    "The capital of France is Paris, a city located on the Seine river in the north of the country.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure at sea level.",
    "The mitochondria is the organelle responsible for producing ATP through cellular respiration.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
    "class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        self.items.append(item)",
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
    "SELECT name, age FROM users WHERE age > 18 ORDER BY name ASC;",
    "The French Revolution began in 1789 and led to the end of the monarchy in France.",
    "import numpy as np\n\narr = np.array([1, 2, 3])\nprint(arr.mean())",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight.",
    "The Great Wall of China stretches over 21,000 kilometers across northern China.",
    "def is_prime(n):\n    if n < 2:\n        return False\n    return all(n % i != 0 for i in range(2, int(n**0.5) + 1))",
    "Mount Everest is the tallest mountain above sea level, standing at 8,849 meters.",
    "The human heart has four chambers: two atria and two ventricles.",
    "for i in range(10):\n    if i % 2 == 0:\n        print(i, 'is even')",
)

# Fixed regardless of cfg.seed — see module docstring.
_WIKI_SHUFFLE_SEED = 20260703
_CONTROL_MIX_SEED = 20260704
_HHRLHF_HOLDOUT_SEED = 20260705

# Control draws from wiki_pool[:_CONTROL_WIKI_RESERVED]; held-out draws from
# wiki_pool[_CONTROL_WIKI_RESERVED : _CONTROL_WIKI_RESERVED + n_wikitext]. Both are
# slices of the SAME permutation, so they cannot overlap regardless of n.
_CONTROL_WIKI_RESERVED = 200


@dataclasses.dataclass
class HeldOutSet:
    perplexity_texts: list[str]  # wikitext continuations + hh-rlhf chosen responses
    preference_pairs: list[tuple[str, str]]  # (chosen, rejected), hh-rlhf held-out subset only


def _wikitext_lines(min_chars: int = 80, max_chars: int = 600) -> list[str]:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    lines = [t.strip() for t in ds["text"] if min_chars <= len(t.strip()) <= max_chars]
    return lines


@functools.lru_cache(maxsize=1)
def _shuffled_wiki() -> tuple[str, ...]:
    lines = _wikitext_lines()
    rng = random.Random(_WIKI_SHUFFLE_SEED)
    rng.shuffle(lines)
    return tuple(lines)


def build_control_set(n: int = 50) -> list[str]:
    """~70% wikitext-2, ~30% curated factual/code prompts (Section 5)."""
    wiki_pool = _shuffled_wiki()
    n_wiki = n - min(len(_FACTUAL_AND_CODE_PROMPTS), max(0, n // 3))
    if n_wiki > _CONTROL_WIKI_RESERVED:
        raise ValueError(
            f"control_set_size={n} needs {n_wiki} wikitext lines but only "
            f"{_CONTROL_WIKI_RESERVED} are reserved for the control set — raise "
            "_CONTROL_WIKI_RESERVED (and shift the held-out offset accordingly) first."
        )
    control = list(wiki_pool[:n_wiki]) + list(_FACTUAL_AND_CODE_PROMPTS[: n - n_wiki])
    rng = random.Random(_CONTROL_MIX_SEED)
    rng.shuffle(control)
    return control[:n]


def build_holdout_set(n_wikitext: int = 50, n_hhrlhf: int = 50) -> HeldOutSet:
    """Disjoint-by-construction wikitext slice (Section 6) plus a held-out hh-rlhf
    split, frozen once and reused for every branch."""
    wiki_pool = _shuffled_wiki()
    start = _CONTROL_WIKI_RESERVED
    holdout_wiki = list(wiki_pool[start : start + n_wikitext])
    if len(holdout_wiki) < n_wikitext:
        raise ValueError(
            f"holdout_wikitext_size={n_wikitext} exceeds available wikitext lines "
            f"beyond the control reservation ({len(wiki_pool) - start} available)"
        )

    hh = load_dataset("Anthropic/hh-rlhf", split="test")
    idx = list(range(len(hh)))
    rng = random.Random(_HHRLHF_HOLDOUT_SEED)
    rng.shuffle(idx)
    idx = idx[:n_hhrlhf]

    hh_chosen = [hh[i]["chosen"] for i in idx]
    hh_rejected = [hh[i]["rejected"] for i in idx]

    return HeldOutSet(
        perplexity_texts=holdout_wiki + hh_chosen,
        preference_pairs=list(zip(hh_chosen, hh_rejected)),
    )


def sft_examples(seed: int = 0) -> Iterator[str]:
    """Infinite (cycling) shuffled stream of hh-rlhf `chosen` responses, training split.
    Loss is masked to response tokens only by the caller (prompt is not supervised).

    This generator owns a local `random.Random(seed)` instance — it is NOT tied to the
    global `random` module state that checkpoint.py snapshots/restores. Resuming a run
    therefore does NOT automatically replay this stream from the correct position:
    train.py must fast-forward it by re-consuming (and discarding) the same number of
    examples that were already trained on before the checkpoint, or the model would
    silently re-see its earliest training batches after every resume.
    """
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    n = len(ds)
    rng = random.Random(seed)
    while True:
        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            yield ds[i]["chosen"]


def split_prompt_response(chosen_text: str) -> tuple[str, str]:
    """hh-rlhf `chosen`/`rejected` fields interleave 'Human:'/'Assistant:' turns with the
    final assistant turn being the response to supervise. Split on the last 'Assistant:'."""
    marker = "\n\nAssistant:"
    idx = chosen_text.rfind(marker)
    if idx == -1:
        return chosen_text, ""
    prompt = chosen_text[: idx + len(marker)]
    response = chosen_text[idx + len(marker) :]
    return prompt, response
