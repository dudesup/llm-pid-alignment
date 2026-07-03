"""Tests for RunConfig validation (pidlora/config.py) — no torch/model dependency."""
from pathlib import Path

import pytest
import yaml

from pidlora.config import RunConfig


def write_yaml(tmp_path, data: dict):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


BASE_FIELDS = dict(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    lora_r=8,
    alpha=16.0,
    seed=0,
    total_steps=1000,
    output_dir="runs/test",
)


class TestRunConfig:
    def test_baseline_is_full_logging(self):
        cfg = RunConfig(run_name="b", branch="baseline", **BASE_FIELDS)
        assert cfg.is_full_logging is True

    def test_sweep_is_not_full_logging(self):
        cfg = RunConfig(run_name="s", branch="sweep", **BASE_FIELDS)
        assert cfg.is_full_logging is False

    def test_reference_cache_defaults_under_output_dir(self):
        cfg = RunConfig(run_name="b", branch="baseline", **BASE_FIELDS)
        assert Path(cfg.reference_logprobs_cache) == Path("runs/test/reference_logprobs.pt")

    def test_invalid_branch_rejected(self):
        with pytest.raises(ValueError, match="branch"):
            RunConfig(run_name="b", branch="heuristic", **BASE_FIELDS)

    def test_new_fields_have_sane_defaults(self):
        cfg = RunConfig(run_name="b", branch="baseline", **BASE_FIELDS)
        assert cfg.grad_clip_max_norm == 1.0
        assert cfg.kl_eval_batch_size == 10
        assert cfg.kl_ema_beta == 0.5

    def test_new_fields_overridable_from_yaml(self, tmp_path):
        data = dict(
            run_name="b", branch="baseline",
            grad_clip_max_norm=2.0, kl_eval_batch_size=5, kl_ema_beta=0.3,
            **BASE_FIELDS,
        )
        path = write_yaml(tmp_path, data)
        cfg = RunConfig.from_yaml(path)
        assert cfg.grad_clip_max_norm == 2.0
        assert cfg.kl_eval_batch_size == 5
        assert cfg.kl_ema_beta == 0.3

    def test_from_yaml_rejects_unknown_field(self, tmp_path):
        data = dict(run_name="b", branch="baseline", **BASE_FIELDS)
        data["totally_unknown_field"] = 123
        path = write_yaml(tmp_path, data)
        with pytest.raises(ValueError, match="Unknown config field"):
            RunConfig.from_yaml(path)

    def test_from_yaml_roundtrip(self, tmp_path):
        data = dict(run_name="b", branch="baseline", **BASE_FIELDS)
        path = write_yaml(tmp_path, data)
        cfg = RunConfig.from_yaml(path)
        assert cfg.run_name == "b"
        assert cfg.branch == "baseline"

    def test_from_yaml_overrides(self, tmp_path):
        data = dict(run_name="b", branch="baseline", **BASE_FIELDS)
        path = write_yaml(tmp_path, data)
        cfg = RunConfig.from_yaml(path, total_steps=42)
        assert cfg.total_steps == 42
