"""Streaming text utilities: sentence chunking, <think> stripping, emotion tags, TTS cleanup."""

from __future__ import annotations

import re
from typing import Iterable, Optional

DEFAULT_EMOTIONS = (
    "neutral",
    "happy",
    "excited",
    "smug",
    "love",
    "sad",
    "angry",
    "scared",
    "confused",
    "thinking",
    "surprised",
    "tired",
)

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "sr", "jr", "vs", "etc", "e.g", "i.e", "no", "approx",
    "prof", "inc", "ltd", "co", "u.s", "u.k", "min", "max", "lt", "sgt", "fig", "vol",
}
_TERMINATORS = ".!?…"
_CLOSERS = "\"')]}’”»"
_CLAUSE_BREAKS = ",;:—–"


class SentenceChunker:
    """Splits a token stream into speakable sentences as early as safely possible.

    ``first_clause_chars`` lets the very first chunk end at a clause break (comma, dash...)
    once it is long enough, so TTS can start before the first full sentence is generated.
    """

    def __init__(self, first_clause_chars: int = 48, max_chars: int = 240) -> None:
        self.buf = ""
        self.first_clause_chars = first_clause_chars
        self.max_chars = max_chars
        self.emitted = 0

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out: list[str] = []
        while True:
            cut = self._find_boundary()
            if cut is None:
                break
            piece, self.buf = self.buf[:cut], self.buf[cut:]
            piece = piece.strip()
            if piece:
                out.append(piece)
                self.emitted += 1
        return out

    def flush(self) -> list[str]:
        piece, self.buf = self.buf.strip(), ""
        if piece:
            self.emitted += 1
            return [piece]
        return []

    def _find_boundary(self) -> Optional[int]:
        buf = self.buf
        n = len(buf)
        i = 0
        while i < n:
            ch = buf[i]
            if ch == "\n":
                if buf[:i].strip():
                    return i + 1
            elif ch in _TERMINATORS:
                j = i
                while j + 1 < n and buf[j + 1] in _TERMINATORS:
                    j += 1
                k = j
                while k + 1 < n and buf[k + 1] in _CLOSERS:
                    k += 1
                if k + 1 >= n:
                    return None  # need the next character to confirm the boundary
                if buf[k + 1].isspace() and not self._is_false_boundary(buf, i, j):
                    return k + 1
                i = k
            i += 1
        if self.emitted == 0 and self.first_clause_chars > 0 and n >= self.first_clause_chars:
            cut = self._clause_break(buf, min_pos=max(12, self.first_clause_chars // 2))
            if cut is not None:
                return cut
        if n > self.max_chars:
            cut = self._clause_break(buf, min_pos=self.max_chars // 2) or buf.rfind(" ", 0, self.max_chars)
            return cut if cut and cut > 0 else self.max_chars
        return None

    @staticmethod
    def _clause_break(buf: str, min_pos: int) -> Optional[int]:
        best = None
        for idx, ch in enumerate(buf):
            if idx >= min_pos and ch in _CLAUSE_BREAKS and idx + 1 < len(buf) and buf[idx + 1].isspace():
                best = idx + 1
        return best

    @staticmethod
    def _is_false_boundary(buf: str, start: int, end: int) -> bool:
        if buf[start] != "." or end != start:
            return False
        match = re.search(r"([A-Za-z][A-Za-z.]*)$", buf[:start])
        if not match:
            return False
        word = match.group(1).lower()
        if word in _ABBREVIATIONS:
            return True
        # Single initials like "J. R. R." are not sentence ends (but "I." and "a." are).
        return (
            len(word) == 1
            and word not in ("i", "a")
            and buf[match.start(1) - 1 : match.start(1)] in (" ", "")
        )


class ThinkFilter:
    """Removes <think>...</think> (or <thinking>) reasoning blocks from a token stream."""

    _OPEN = re.compile(r"<think(?:ing)?>", re.IGNORECASE)
    _CLOSE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)
    _MAX_TAG = len("</thinking>")

    def __init__(self) -> None:
        self.buf = ""
        self.in_think = False

    def feed(self, text: str) -> str:
        self.buf += text
        out: list[str] = []
        while self.buf:
            if self.in_think:
                m = self._CLOSE.search(self.buf)
                if not m:
                    self.buf = self.buf[-(self._MAX_TAG - 1):]
                    break
                self.buf = self.buf[m.end():]
                self.in_think = False
                continue
            m_open = self._OPEN.search(self.buf)
            m_close = self._CLOSE.search(self.buf)
            if m_close and (not m_open or m_close.start() < m_open.start()):
                out.append(self.buf[: m_close.start()])
                self.buf = self.buf[m_close.end():]
                continue
            if m_open:
                out.append(self.buf[: m_open.start()])
                self.buf = self.buf[m_open.end():]
                self.in_think = True
                continue
            keep = self._partial_tag_suffix(self.buf)
            out.append(self.buf[: len(self.buf) - keep])
            self.buf = self.buf[len(self.buf) - keep:]
            break
        return "".join(out)

    def flush(self) -> str:
        rest, self.buf = ("" if self.in_think else self.buf), ""
        self.in_think = False
        return rest

    @classmethod
    def _partial_tag_suffix(cls, text: str) -> int:
        idx = text.rfind("<")
        if idx == -1 or len(text) - idx >= cls._MAX_TAG:
            return 0
        tail = text[idx:].lower()
        for tag in ("<think>", "<thinking>", "</think>", "</thinking>"):
            if tag.startswith(tail):
                return len(text) - idx
        return 0


_TAG_RE = re.compile(r"\[([A-Za-z][A-Za-z _-]{0,24})\]")
_ACTION_RE = re.compile(r"\*([^*\n]{1,60})\*")
_ACTION_EMOTIONS = {
    "laugh": "happy", "giggle": "happy", "chuckle": "happy", "grin": "smug", "smirk": "smug",
    "sigh": "tired", "gasp": "surprised", "cry": "sad", "sob": "sad", "pout": "sad",
    "blush": "love", "growl": "angry", "hmm": "thinking", "yawn": "tired",
}


def extract_tags(text: str, vocabulary: Iterable[str] = DEFAULT_EMOTIONS) -> tuple[str, list[str]]:
    """Pull [emotion] tags and *actions* out of a sentence. Returns (spoken_text, emotions)."""
    vocab = {v.lower() for v in vocabulary}
    emotions: list[str] = []

    def tag_sub(m: re.Match) -> str:
        name = m.group(1).strip().lower().replace(" ", "_")
        if name in vocab:
            emotions.append(name)
        return " "

    def action_sub(m: re.Match) -> str:
        words = m.group(1).lower()
        for key, emo in _ACTION_EMOTIONS.items():
            if key in words and emo in vocab:
                emotions.append(emo)
                break
        return " "

    text = _TAG_RE.sub(tag_sub, text)
    text = _ACTION_RE.sub(action_sub, text)
    return re.sub(r"\s+", " ", text).strip(), emotions


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\u200d\ufe0f]+", re.UNICODE
)


def clean_for_tts(text: str) -> str:
    text = _URL_RE.sub("a link", text)
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"`+", "", text)
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    text = re.sub(r"^\s*(#+|[-•]|\d+[.)])\s+", "", text)
    text = text.replace("*", "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def strip_speaker_prefix(text: str, names: Iterable[str]) -> str:
    """Models sometimes start with 'Nexa: ...'; the character should not read her own name tag."""
    for name in names:
        if not name:
            continue
        m = re.match(rf"^\s*\(?{re.escape(name)}\)?\s*[:：]\s*", text, re.IGNORECASE)
        if m:
            return text[m.end():]
    return text


def has_speakable_content(text: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9À-ɏ぀-ヿ一-鿿]", text))
