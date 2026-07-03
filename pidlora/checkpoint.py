"""Checkpoint + resume. output_dir is expected to be Drive-mounted in Colab so state
survives a session disconnect; /content itself does not (run_colab.ipynb notes).

maybe_resume returns start_step=0 when nothing is found on disk, so the launcher can
always pass --resume unconditionally: a fresh run and a resumed run take the same code
path, and an explicit --no-resume is the only way to discard existing progress.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors


def _checkpoints_dir(output_dir: str) -> Path:
    return Path(output_dir) / "checkpoints"


def _step_dir(output_dir: str, step: int) -> Path:
    return _checkpoints_dir(output_dir) / f"step_{step:06d}"


def save_checkpoint(output_dir: str, step: int, model, optimizer, extra_state: Optional[dict] = None) -> None:
    """extra_state: opaque dict for state that doesn't belong to the model/optimizer —
    e.g. the KL EMA filter's {beta, value} — round-tripped verbatim through resume."""
    step_dir = _step_dir(output_dir, step)
    step_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(step_dir / "adapter")

    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "extra": extra_state or {},
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, step_dir / "trainer_state.pt")

    latest_path = _checkpoints_dir(output_dir) / "latest.txt"
    latest_path.write_text(f"step_{step:06d}", encoding="utf-8")


def _latest_step_dir(output_dir: str) -> Optional[Path]:
    latest_path = _checkpoints_dir(output_dir) / "latest.txt"
    if not latest_path.exists():
        return None
    name = latest_path.read_text(encoding="utf-8").strip()
    step_dir = _checkpoints_dir(output_dir) / name
    return step_dir if step_dir.exists() else None


def maybe_resume(output_dir: str, model, optimizer, resume: bool = True) -> tuple[int, dict]:
    """Returns (start_step, extra_state). start_step=0 and extra_state={} when nothing
    is found on disk or resume=False."""
    if not resume:
        return 0, {}
    step_dir = _latest_step_dir(output_dir)
    if step_dir is None:
        return 0, {}

    # The adapter already exists on `model` (created by get_peft_model) — we're
    # restoring its weights in place, not attaching a new adapter, so we load the
    # state dict directly rather than using PeftModel.load_adapter.
    adapter_weights_path = step_dir / "adapter" / "adapter_model.safetensors"
    adapter_state = load_safetensors(str(adapter_weights_path))
    set_peft_model_state_dict(model, adapter_state, adapter_name="default")

    state = torch.load(step_dir / "trainer_state.pt", map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])

    random.setstate(state["rng_python"])
    np.random.set_state(state["rng_numpy"])
    torch.set_rng_state(state["rng_torch"])
    if torch.cuda.is_available() and state.get("rng_torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["rng_torch_cuda"])

    return int(state["step"]), state.get("extra", {})
