"""JSONL transcripts: the raw material for the self-improvement loop (training/curate.py)."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class TranscriptLogger:
    def __init__(self, folder: str | Path, session_id: str) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / f"{session_id}.jsonl"
        self.session_id = session_id
        self.count = 0

    def log(self, record: dict[str, Any]) -> None:
        record = {"ts": time.time(), "session": self.session_id, **record}
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self.count += 1
        except OSError as exc:
            log.warning("could not write transcript: %s", exc)

    @staticmethod
    def fingerprint(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
