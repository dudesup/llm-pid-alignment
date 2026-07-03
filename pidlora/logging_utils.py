"""Append-only JSONL metrics log. Flushed on every write so that a Colab disconnect
loses at most the in-flight step, not the whole run — output_dir is expected to be on
Drive, which is why this matters (Section: run_colab.ipynb launcher notes)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class MetricsLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, step: int, event: str, **fields: Any) -> None:
        record = {"step": step, "event": event, "time": time.time(), **fields}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_metrics_jsonl(path: str | Path) -> list[dict]:
    """Reads the log and deduplicates keep-last by (event, step). A resume can re-log
    the same (event, step) pair after a Colab disconnect (the step that was in flight
    when the session died); without this, every plot built on this function would
    zigzag between the stale pre-disconnect value and the real one at that step."""
    seen: dict[tuple, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = (record.get("event"), record.get("step"))
            seen[key] = record
    return sorted(seen.values(), key=lambda r: (r.get("step", 0)))
