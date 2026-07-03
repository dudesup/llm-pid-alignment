"""Tests for MetricsLogger / read_metrics_jsonl (pidlora/logging_utils.py) — no
torch/model dependency."""
import json

from pidlora.logging_utils import MetricsLogger, read_metrics_jsonl


class TestMetricsLogger:
    def test_writes_one_json_line_per_call(self, tmp_path):
        path = tmp_path / "metrics.jsonl"
        with MetricsLogger(path) as logger:
            logger.log(step=0, event="a", x=1)
            logger.log(step=1, event="b", x=2)

        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["x"] == 1


class TestReadMetricsJsonlDedup:
    def test_keeps_last_value_for_duplicate_event_step(self, tmp_path):
        """Simulates a resume that re-logs kl_eval at the same step after a disconnect
        — the stale pre-disconnect value must not survive into the returned records."""
        path = tmp_path / "metrics.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 25, "event": "kl_eval", "kl_raw": 0.11}) + "\n")
            f.write(json.dumps({"step": 50, "event": "kl_eval", "kl_raw": 0.22}) + "\n")
            # session died and resumed from a checkpoint before step 50's write landed —
            # step 50 gets re-logged with the value that actually happened
            f.write(json.dumps({"step": 50, "event": "kl_eval", "kl_raw": 0.23}) + "\n")

        records = read_metrics_jsonl(path)
        kl_at_50 = [r for r in records if r["event"] == "kl_eval" and r["step"] == 50]
        assert len(kl_at_50) == 1
        assert kl_at_50[0]["kl_raw"] == 0.23

    def test_distinct_events_at_same_step_both_kept(self, tmp_path):
        path = tmp_path / "metrics.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 25, "event": "train_step", "loss": 1.0}) + "\n")
            f.write(json.dumps({"step": 25, "event": "kl_eval", "kl_raw": 0.5}) + "\n")

        records = read_metrics_jsonl(path)
        assert len(records) == 2

    def test_output_sorted_by_step(self, tmp_path):
        path = tmp_path / "metrics.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 50, "event": "a"}) + "\n")
            f.write(json.dumps({"step": 0, "event": "b"}) + "\n")
            f.write(json.dumps({"step": 25, "event": "c"}) + "\n")

        records = read_metrics_jsonl(path)
        assert [r["step"] for r in records] == [0, 25, 50]

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "metrics.jsonl"
        path.write_text('{"step": 0, "event": "a"}\n\n{"step": 1, "event": "b"}\n', encoding="utf-8")
        records = read_metrics_jsonl(path)
        assert len(records) == 2
