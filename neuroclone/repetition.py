"""Loop prevention: drop near-duplicate sentences and surface overused phrases to the prompt."""

from __future__ import annotations

from collections import Counter, deque

from .safety.filter import normalize

_STOP = set("a an the and or but to of in on is it i you me my we us that this so just be are was".split())


def _grams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = normalize(text).split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class RepetitionGuard:
    def __init__(self, window: int = 30, threshold: float = 0.6) -> None:
        self.recent: deque[str] = deque(maxlen=window)
        self.threshold = threshold

    def is_repetitive(self, text: str) -> bool:
        norm = normalize(text)
        if not norm:
            return False
        words = norm.split()
        if len(words) < 4:
            # Short interjections ("Let's go!") may recur, just not back to back.
            return any(normalize(prev) == norm for prev in list(self.recent)[-3:])
        grams = _grams(text)
        return any(jaccard(grams, _grams(prev)) >= self.threshold for prev in self.recent)

    def record(self, text: str) -> None:
        if normalize(text):
            self.recent.append(text)

    def overused_phrases(self, min_count: int = 3, top: int = 4) -> list[str]:
        counts: Counter[str] = Counter()
        for line in self.recent:
            words = normalize(line).split()
            seen = set()
            for n in (3, 4):
                for i in range(len(words) - n + 1):
                    gram = words[i : i + n]
                    if sum(w not in _STOP for w in gram) < 2:
                        continue
                    phrase = " ".join(gram)
                    if phrase not in seen:
                        seen.add(phrase)
                        counts[phrase] += 1
        frequent = [p for p, c in counts.most_common() if c >= min_count]
        result: list[str] = []
        for phrase in frequent:
            if not any(phrase in kept or kept in phrase for kept in result):
                result.append(phrase)
            if len(result) >= top:
                break
        return result
