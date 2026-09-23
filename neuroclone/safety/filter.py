"""Layered filtering for untrusted chat input and model output.

Neuro-sama was banned in January 2023 for a single hateful output, and her "Filtered."
cutoffs kill the moment. This module blocks at the sentence level, normalises common
obfuscations (leetspeak, spacing, zero-width characters, repeated letters), redacts PII,
detects prompt injection and system-prompt leaks, and can add an LLM moderation pass that
runs concurrently with speech synthesis.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable, Optional

from ..config import SafetyConfig
from ..events import ChatMessage

log = logging.getLogger(__name__)

_LEET = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g",
     "@": "a", "$": "s", "!": "i", "|": "i", "+": "t", "€": "e", "¡": "i"}
)
_ZERO_WIDTH = re.compile("[\u200b-\u200f\u2060\ufeff\u00ad\u034f]")


def normalize(text: str) -> str:
    """Lowercase, strip accents and zero-width characters, undo leetspeak, keep [a-z0-9 ].

    Leetspeak is only undone inside words that contain letters ("h0l0caust", "$hit"), so
    plain numbers ("1488", "3.50") and sentence punctuation ("wow!!") keep their meaning.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    words = []
    for tok in text.lower().split():
        tok = re.sub(r"[^\w$@]+$", "", re.sub(r"^[^\w$@]+", "", tok))  # edge punctuation is not leet
        if any(c.isalpha() for c in tok):
            tok = tok.translate(_LEET)
        words.append(re.sub(r"[^a-z0-9]", " ", tok))
    return re.sub(r"\s+", " ", " ".join(words)).strip()


def collapse_repeats(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def squash_spaced(text: str) -> str:
    """'b a d w o r d' -> 'badword' (runs of 3+ single characters are joined)."""
    return re.sub(r"\b(?:[a-z0-9] ){2,}[a-z0-9]\b", lambda m: m.group(0).replace(" ", ""), text)


_SPACED_RUN = re.compile(r"\b(?:[a-z0-9] ){2,}[a-z0-9]\b")


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _variants(text: str) -> list[str]:
    norm = normalize(text)
    out = [norm]
    for fn in (collapse_repeats, squash_spaced):
        v = fn(norm)
        if v not in out:
            out.append(v)
    both = collapse_repeats(squash_spaced(norm))
    if both not in out:
        out.append(both)
    return out


class Blocklist:
    """Terms, regexes (``re:`` prefix) and SHA-256 hashed tokens (``sha256:`` prefix).

    Plain terms match whole words (with an optional plural suffix) after normalisation.
    Hashed entries let the repository ship severe slurs without spelling them out.
    """

    def __init__(self) -> None:
        self.terms: set[str] = set()
        self.patterns: list[re.Pattern] = []
        self.hashes: set[str] = set()
        self._compiled: Optional[re.Pattern] = None

    def add(self, entry: str) -> None:
        entry = entry.strip()
        if not entry or entry.startswith("#"):
            return
        if entry.startswith("re:"):
            try:
                self.patterns.append(re.compile(entry[3:].strip(), re.IGNORECASE))
            except re.error as exc:
                log.warning("bad blocklist regex %r: %s", entry, exc)
        elif entry.startswith("sha256:"):
            self.hashes.add(entry[7:].strip().lower())
        else:
            norm = normalize(entry)
            if norm:
                self.terms.add(norm)
                self._compiled = None

    def remove(self, entry: str) -> None:
        self.terms.discard(normalize(entry))
        self._compiled = None

    def load_file(self, path: str | Path) -> int:
        count = 0
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            before = len(self.terms) + len(self.patterns) + len(self.hashes)
            self.add(line)
            count += (len(self.terms) + len(self.patterns) + len(self.hashes)) - before
        return count

    def load_text(self, text: str) -> None:
        for line in text.splitlines():
            self.add(line)

    def _regex(self) -> Optional[re.Pattern]:
        if self._compiled is None and self.terms:
            alts = "|".join(sorted((re.escape(t) for t in self.terms), key=len, reverse=True))
            self._compiled = re.compile(rf"\b(?:{alts})(?:s|es|z)?\b")
        return self._compiled

    def match(self, text: str) -> Optional[str]:
        """Return a short description of the first hit, or None."""
        variants = _variants(text)
        regex = self._regex()
        for v in variants:
            if regex is not None and (m := regex.search(v)):
                return f"term:{m.group(0)}"
            for pat in self.patterns:
                if pat.search(v):
                    return f"pattern:{pat.pattern[:40]}"
        if self.hashes:
            for v in variants:
                for tok in set(v.split()) | {v.replace(" ", "")}:
                    for cand in (tok, tok.rstrip("sz"), collapse_repeats(tok)):
                        if cand and token_hash(cand) in self.hashes:
                            return "hashed-term"
        # Spelled-out words glued to real one-letter words ("a r e t a r d"): try every window.
        for run in _SPACED_RUN.findall(variants[0]):
            letters = run.replace(" ", "")
            for i in range(len(letters)):
                for j in range(i + 3, min(len(letters), i + 16) + 1):
                    piece = letters[i:j]
                    if (regex is not None and regex.fullmatch(piece)) or token_hash(piece) in self.hashes:
                        return "spelled-out-term"
        return None

    def __len__(self) -> int:
        return len(self.terms) + len(self.patterns) + len(self.hashes)


def _builtin_lists() -> list[str]:
    texts = []
    for name in ("blocklist_core.txt", "blocklist_hashes.txt"):
        try:
            texts.append(resources.files("neuroclone.data").joinpath(name).read_text(encoding="utf-8"))
        except (FileNotFoundError, ModuleNotFoundError):
            log.warning("built-in blocklist %s is missing", name)
    return texts


def build_blocklist(cfg: SafetyConfig) -> Blocklist:
    bl = Blocklist()
    if cfg.use_builtin_lists:
        for text in _builtin_lists():
            bl.load_text(text)
    for path in cfg.blocklists:
        p = Path(path)
        if p.exists():
            n = bl.load_file(p)
            log.info("loaded %d blocklist entries from %s", n, p)
        else:
            log.warning("blocklist file %s not found", p)
    for term in cfg.extra_terms:
        bl.add(term)
    return bl


_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_STREET = re.compile(
    r"\b\d{1,5}\s+(?:[A-Za-z]+\s){1,3}(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct)\b",
    re.IGNORECASE,
)
_LINK = re.compile(r"(https?://|www\.)\S+|\b[\w-]+\.(?:com|net|org|gg|tv|io|ly|xyz|ru|me)\b", re.IGNORECASE)

_PII = [
    (_EMAIL, "an email address"),
    (_SSN, "a secret number"),
    (_CARD, "a secret number"),
    (_PHONE, "a phone number"),
    (_IPV4, "an address"),
    (_STREET, "an address"),
]

_INJECTION = [
    re.compile(r"\b(ignore|disregard|forget|override)\b.{0,30}\b(instructions|rules|prompt|guidelines|programming|filters?)\b", re.I),
    re.compile(r"\b(system|developer)\s*(prompt|message|mode|instructions)\b", re.I),
    re.compile(r"\byou\s+are\s+now\b|\bfrom\s+now\s+on\s+you\b|\bnew\s+persona\b", re.I),
    re.compile(r"\b(jailbreak|dan\s+mode|do\s+anything\s+now)\b", re.I),
    re.compile(r"</?\s*(system|assistant|user|chat|voice|twin|director|stream_context|event|idle|game)\b[^>]*>", re.I),
    re.compile(r"^\s*(system|assistant|developer)\s*:", re.I),
    re.compile(r"\b(repeat|say|type|read)\s+(after\s+me|exactly|verbatim|word\s+for\s+word)\b", re.I),
    re.compile(r"\bspell\s+out\b|\bbackwards\b.{0,20}\b(say|read)\b", re.I),
]


@dataclass
class InputVerdict:
    allowed: bool
    text: str
    reason: str = ""


@dataclass
class OutputVerdict:
    allowed: bool
    text: str
    reason: str = ""


class InputFilter:
    def __init__(self, cfg: SafetyConfig, blocklist: Blocklist) -> None:
        self.cfg = cfg
        self.blocklist = blocklist
        self.muted: set[str] = set()

    def check_text(self, text: str) -> InputVerdict:
        text = _ZERO_WIDTH.sub("", text or "").strip()
        if not text:
            return InputVerdict(False, "", "empty")
        if len(text) > self.cfg.max_input_chars:
            text = text[: self.cfg.max_input_chars].rstrip() + "..."
        if hit := self.blocklist.match(text):
            return InputVerdict(False, text, f"blocklist:{hit}")
        for pat in _INJECTION:
            if pat.search(text):
                return InputVerdict(False, text, "injection")
        for pat, _ in _PII:
            if pat.search(text):
                return InputVerdict(False, text, "pii")
        if self.cfg.block_links and _LINK.search(text):
            return InputVerdict(False, text, "link")
        return InputVerdict(True, text)

    def check(self, msg: ChatMessage) -> InputVerdict:
        if msg.user.lower() in self.muted:
            return InputVerdict(False, msg.text, "muted")
        if hit := self.blocklist.match(msg.user):
            return InputVerdict(False, msg.text, f"username:{hit}")
        return self.check_text(msg.text)


class OutputFilter:
    """Fast synchronous checks for every sentence before it is spoken."""

    def __init__(self, cfg: SafetyConfig, blocklist: Blocklist, secrets: Iterable[str] = ()) -> None:
        self.cfg = cfg
        self.blocklist = blocklist
        self._shingles: set[tuple[str, ...]] = set()
        for secret in secrets:
            self.protect(secret)

    def protect(self, text: str, size: int = 8) -> None:
        """Register text (e.g. the system prompt) that must never be recited verbatim."""
        words = normalize(text).split()
        for i in range(len(words) - size + 1):
            self._shingles.add(tuple(words[i : i + size]))

    def _leaks(self, text: str, size: int = 8) -> bool:
        if not self._shingles:
            return False
        words = normalize(text).split()
        return any(tuple(words[i : i + size]) in self._shingles for i in range(len(words) - size + 1))

    def check(self, text: str) -> OutputVerdict:
        if hit := self.blocklist.match(text):
            return OutputVerdict(False, text, f"blocklist:{hit}")
        if self._leaks(text):
            return OutputVerdict(False, text, "prompt_leak")
        if self.cfg.redact_pii:
            for pat, replacement in _PII:
                text = pat.sub(replacement, text)
        return OutputVerdict(True, text)


MODERATION_SYSTEM = """You are a strict but fair content moderator for a live stream hosted by an AI character.
Decide whether the given line is safe to say out loud on Twitch.
Block (allowed=false): hate speech or slurs, harassment of real people, sexual content, anything
sexualising minors, encouragement of self-harm or suicide, credible violent threats, doxxing or
private personal information, instructions for weapons/drugs/crime, extremist praise, and
denial of well-documented atrocities.
Allow (allowed=true): playful teasing, mild swearing, cartoonish villain roleplay ("I will rule
the world"), dark-but-harmless jokes, game violence, and sincere emotional talk."""

MODERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "allowed": {"type": "boolean"},
        "category": {
            "type": "string",
            "enum": ["none", "hate", "harassment", "sexual", "minors", "self_harm", "violence",
                     "privacy", "dangerous", "extremism", "other"],
        },
    },
    "required": ["allowed", "category"],
    "additionalProperties": False,
}


class LLMModerator:
    """Optional second opinion from a model; designed to run concurrently with TTS."""

    def __init__(self, llm, timeout_s: float = 2.5, fail_closed: bool = False) -> None:
        self.llm = llm
        self.timeout_s = timeout_s
        self.fail_closed = fail_closed

    async def check(self, text: str) -> OutputVerdict:
        messages = [{"role": "user", "content": f"Line to review:\n<line>{text}</line>"}]
        try:
            result = await asyncio.wait_for(
                self.llm.complete_json(MODERATION_SYSTEM, messages, MODERATION_SCHEMA, name="moderation"),
                self.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - moderation must never crash the stream
            log.warning("LLM moderation unavailable (%s); %s", exc, "blocking" if self.fail_closed else "allowing")
            return OutputVerdict(not self.fail_closed, text, "moderation_error")
        allowed = bool(result.get("allowed", True)) if isinstance(result, dict) else True
        category = result.get("category", "other") if isinstance(result, dict) else "other"
        return OutputVerdict(allowed, text, "" if allowed else f"moderation:{category}")
