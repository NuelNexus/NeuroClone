import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(coro, timeout: float = 30.0):
    """Run a coroutine in a fresh event loop with a safety timeout."""
    return asyncio.run(asyncio.wait_for(coro, timeout))


@pytest.fixture
def tmp_cfg(tmp_path):
    """A mock-profile config writing all runtime data under tmp_path."""
    from neuroclone.config import Config, apply_mock_profile

    cfg = apply_mock_profile(Config())
    cfg.llm.mock_delay_s = 0.0
    cfg.llm.seed = 7
    cfg.audio.time_scale = 0.0
    cfg.memory.path = str(tmp_path / "memory.sqlite3")
    cfg.logging.transcripts_dir = str(tmp_path / "transcripts")
    cfg.games.port = 0
    cfg.overlay.port = 0
    cfg.conductor.reply_gap_s = 0.0
    cfg.conductor.event_batch_s = 0.0
    cfg.conductor.idle_after_s = 0.0
    return cfg
