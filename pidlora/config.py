"""Run configuration schema. Every field that affects a result lives here or downstream
in the package — the launcher notebook only ever passes a config path plus --resume/--output-dir.

Scope note: only static-alpha branches (baseline, sweep) are implemented right now. The
controller (PI) and threshold-heuristic branches come later as a separate addition —
this schema deliberately has no dynamic-alpha / controller fields yet.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal, Optional

import yaml

Branch = Literal["baseline", "sweep"]
_VALID_BRANCHES = ("baseline", "sweep")


@dataclasses.dataclass
class RunConfig:
    run_name: str
    branch: Branch

    # Model / adapter (Section 5, 7)
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    lora_r: int = 8
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    alpha: float = 16.0  # static for the run's whole duration
    use_4bit: bool = True  # False for CPU smoke tests — bitsandbytes 4-bit needs CUDA;
                            # real T4 runs must keep this True (Section 5 VRAM budget)

    # Data (Section 5, 6)
    seed: int = 0
    control_set_size: int = 50
    holdout_wikitext_size: int = 50
    holdout_hhrlhf_size: int = 50
    max_seq_len: int = 512
    topk_logprobs: int = 1000  # tail-handling truncation (Section 5)

    # Training (Section 15)
    total_steps: int = 1000
    batch_size: int = 4
    grad_accum_steps: int = 4
    learning_rate: float = 2e-4
    grad_clip_max_norm: float = 1.0

    # Measurement / logging cadence (Section 7, 9)
    kl_eval_every: int = 25  # logged on ALL branches, incl. baseline (Figure 1 density)
    kl_eval_batch_size: int = 10  # control-set mini-batch size for KL forward passes
    kl_ema_beta: float = 0.5  # smoothing for the kl_filt field logged alongside kl_raw
    holdout_eval_every: int = 200  # skipped entirely for branch == "sweep" (end-of-run only)
    holdout_eval_batch_size: int = 4  # held-out perplexity mini-batch size
    checkpoint_every: int = 250

    # Paths — output_dir should be a Drive-mounted path in Colab so it survives a disconnect
    output_dir: str = "runs/default"
    reference_logprobs_cache: Optional[str] = None  # defaults to f"{output_dir}/reference_logprobs.pt"

    def __post_init__(self) -> None:
        if self.branch not in _VALID_BRANCHES:
            raise ValueError(f"branch must be one of {_VALID_BRANCHES}, got {self.branch!r}")
        if self.reference_logprobs_cache is None:
            self.reference_logprobs_cache = str(Path(self.output_dir) / "reference_logprobs.pt")

    @property
    def is_full_logging(self) -> bool:
        """Sweep branches use end-of-run metrics only (Section 15) — everyone else logs periodically."""
        return self.branch != "sweep"

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        raw.update(overrides)
        if "lora_target_modules" in raw and isinstance(raw["lora_target_modules"], list):
            raw["lora_target_modules"] = tuple(raw["lora_target_modules"])
        known_fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known_fields
        if unknown:
            raise ValueError(f"Unknown config field(s) in {path}: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
