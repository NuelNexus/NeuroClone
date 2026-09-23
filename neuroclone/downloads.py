"""One-time downloads for offline use: Kokoro voice files, Whisper models, Ollama model sizes."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

Progress = Callable[[str, int, int], None]  # (label, done_bytes, total_bytes)

KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/"
# name -> (sha256, size in bytes) of the files we tested.
KOKORO_FILES = {
    "kokoro-v1.0.onnx": ("beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a", 325_505_369),
    "voices-v1.0.bin": ("bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d", 28_214_398),
}


def download_file(url: str, dest: Path, *, sha256: str = "", progress: Optional[Progress] = None,
                  label: str = "") -> Path:
    """Download to ``dest`` atomically (``.part`` then rename), optionally checking a SHA-256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "neuroclone-setup"})
    with urllib.request.urlopen(request, timeout=60) as resp, open(part, "wb") as out:  # noqa: S310 - fixed URLs
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if progress is not None:
                progress(label or dest.name, done, total)
    if sha256 and digest.hexdigest() != sha256:
        log.warning("%s: checksum differs from the tested file (upstream may have re-exported it)", dest.name)
    os.replace(part, dest)
    return dest


def file_ok(path: Path, size: int = 0) -> bool:
    return path.exists() and (not size or path.stat().st_size == size)


def ensure_kokoro(model_path: str, voices_path: str, progress: Optional[Progress] = None) -> list[Path]:
    """Fetch the Kokoro model and voices if they are missing. Returns the files downloaded."""
    fetched = []
    for target, name in ((Path(model_path), "kokoro-v1.0.onnx"), (Path(voices_path), "voices-v1.0.bin")):
        sha, size = KOKORO_FILES[name]
        if target.name != name:  # a custom file (e.g. the fp16 model): only check it exists
            sha, size = "", 0
        if file_ok(target, size):
            continue
        fetched.append(download_file(KOKORO_BASE + target.name, target, sha256=sha, progress=progress,
                                     label=f"Kokoro {target.name}"))
    return fetched


def ensure_whisper(model: str, cache_dir: str) -> str:
    """Download a faster-whisper model into ``cache_dir`` (where the runtime will look for it)."""
    from faster_whisper.utils import download_model

    return download_model(model, cache_dir=cache_dir or None)


def registry_size_gb(tag: str, timeout: float = 10.0) -> Optional[float]:
    """Exact download size of an Ollama library model, from its registry manifest (None if unknown)."""
    name, _, version = tag.partition(":")
    if "/" not in name:
        name = f"library/{name}"
    url = f"https://registry.ollama.ai/v2/{name}/manifests/{version or 'latest'}"
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json",
                                                   "User-Agent": "neuroclone-setup"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310 - fixed host
            manifest = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - offline, blocked, or renamed: fall back to estimates
        log.debug("registry lookup for %s failed: %s", tag, exc)
        return None
    layers = manifest.get("layers") or []
    total = sum(int(layer.get("size") or 0) for layer in layers)
    return total / 1e9 if total else None


class ProgressBar:
    """A tiny single-line progress printer (no dependencies, works in the Windows console)."""

    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stdout
        self._last = 0.0
        self._label = ""

    def __call__(self, label: str, done: int, total: int) -> None:
        now = time.monotonic()
        if label == self._label and now - self._last < 0.2 and (not total or done < total):
            return
        self._label, self._last = label, now
        if total:
            pct = 100 * done / total
            bar = "#" * int(pct / 4)
            text = f"\r  {label[:38]:<38} [{bar:<25}] {pct:5.1f}% of {total / 1e9:.2f} GB"
        elif done:
            text = f"\r  {label[:38]:<38} {done / 1e6:8.1f} MB"
        else:  # a status step such as "verifying sha256 digest"
            text = f"\r  {label[:70]:<70}"
        self.stream.write(text)
        self.stream.flush()

    def done(self, message: str = "") -> None:
        self.stream.write("\r" + " " * 90 + "\r")
        if message:
            self.stream.write(message + "\n")
        self.stream.flush()
        self._label = ""
