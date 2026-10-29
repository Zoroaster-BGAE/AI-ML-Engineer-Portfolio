"""Verbatim guard (FR-14): synthesis output is checked for long word-for-word overlap with the retrieved
chunks, so the assistant cannot become a manual-dumping channel. Applied per sentence while streaming."""
from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

_WORD = re.compile(r"\w+", re.UNICODE)
_BOUNDARY = re.compile(r"(?<=[.!?;:])\s+|\n+")
_MARKER = re.compile(r"[ \t]?\[(\d{1,2})\]")


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def ngram_set(texts: Iterable[str], n: int) -> set[tuple[str, ...]]:
    grams: set[tuple[str, ...]] = set()
    for t in texts:
        w = _words(t)
        grams.update(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return grams


def overlaps(sentence: str, grams: set[tuple[str, ...]], n: int) -> bool:
    w = _words(sentence)
    return any(tuple(w[i:i + n]) in grams for i in range(len(w) - n + 1))


class SentenceGuard:
    """Feed streamed deltas; get back guarded text pieces. Collects [n] citation markers in `.cited` (and strips
    them unless `keep_markers`) and drops any sentence that copies >= n consecutive words from a source chunk."""

    def __init__(self, source_texts: Iterable[str], n: int, keep_markers: bool = False):
        self.n = n
        self.grams = ngram_set(source_texts, n)
        self.buf = ""
        self.cited: list[int] = []
        self.dropped = 0
        self.keep_markers = keep_markers

    def _clean(self, seg: str) -> str:
        for m in _MARKER.finditer(seg):
            k = int(m.group(1))
            if k not in self.cited:
                self.cited.append(k)
        return seg if self.keep_markers else _MARKER.sub("", seg)

    def _emit(self, seg: str) -> str:
        if seg.strip() and overlaps(seg, self.grams, self.n):
            self.dropped += 1
            return ""
        return self._clean(seg)

    def feed(self, delta: str) -> Iterator[str]:
        self.buf += delta
        last = 0
        for m in _BOUNDARY.finditer(self.buf):
            seg = self.buf[last:m.end()]
            last = m.end()
            if out := self._emit(seg):
                yield out
        self.buf = self.buf[last:]

    def flush(self) -> Iterator[str]:
        if self.buf:
            out = self._emit(self.buf)
            self.buf = ""
            if out:
                yield out
