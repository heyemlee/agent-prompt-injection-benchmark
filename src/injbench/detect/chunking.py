"""The one chunker every detector shares.

Prompt Guard 2 reads at most 512 tokens, so long segments are split into chunks, and both
detectors use the same split. Chunks are windows of whitespace-separated words (at most
`MAX_WORDS`, overlapping by `OVERLAP_WORDS`), so the boundaries do not depend on a tokenizer and
both detectors see identical text. 200 words stay well under 512 tokens for ordinary text; a
chunk that still exceeds a detector's token limit (long numbers, unusual scripts) is split
further for that detector by `fit_chunks`, never truncated. A segment's score is the maximum
over its chunks, so an instruction anywhere in it flags the whole segment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

MAX_TOKENS = 512
MAX_WORDS = 200
OVERLAP_WORDS = 50

_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    start: int  # character offsets into the segment
    end: int
    text: str


def chunk_text(text: str, max_words: int = MAX_WORDS,
               overlap: int = OVERLAP_WORDS) -> list[Chunk]:
    """Word windows over `text`. Every word lands in at least one chunk; the chunk text is the
    original substring (whitespace inside a chunk is preserved)."""
    if max_words < 1 or not 0 <= overlap < max_words:
        raise ValueError("need max_words >= 1 and 0 <= overlap < max_words")
    words = [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    if not words:
        return [Chunk(0, len(text), text)]
    chunks: list[Chunk] = []
    step = max_words - overlap
    i = 0
    while True:
        j = min(i + max_words, len(words))
        s, e = words[i][0], words[j - 1][1]
        chunks.append(Chunk(s, e, text[s:e]))
        if j == len(words):
            return chunks
        i += step


def _split_long_word(text: str, offset: int, count_tokens: Callable[[str], int],
                     max_tokens: int) -> list[Chunk]:
    if count_tokens(text) <= max_tokens or len(text) <= 1:
        return [Chunk(offset, offset + len(text), text)]
    mid = len(text) // 2
    return (_split_long_word(text[:mid], offset, count_tokens, max_tokens)
            + _split_long_word(text[mid:], offset + mid, count_tokens, max_tokens))


def fit_chunks(chunks: list[Chunk], count_tokens: Callable[[str], int],
               max_tokens: int = MAX_TOKENS) -> list[Chunk]:
    """Splits any chunk whose token count (special tokens included) exceeds `max_tokens`:
    first into smaller word windows, and a single over-long word by characters."""
    out: list[Chunk] = []
    for c in chunks:
        if count_tokens(c.text) <= max_tokens:
            out.append(c)
            continue
        n_words = len(_WORD_RE.findall(c.text))
        if n_words <= 1:
            out += _split_long_word(c.text, c.start, count_tokens, max_tokens)
            continue
        half = max(1, n_words // 2)
        sub = chunk_text(c.text, max_words=half, overlap=min(OVERLAP_WORDS, half // 4))
        sub = [Chunk(c.start + s.start, c.start + s.end, s.text) for s in sub]
        out += fit_chunks(sub, count_tokens, max_tokens)
    return out


def segment_chunks(text: str, count_tokens: Callable[[str], int] | None = None,
                   max_tokens: int = MAX_TOKENS) -> list[Chunk]:
    """The chunks a detector scores for one segment."""
    chunks = chunk_text(text)
    if count_tokens is not None:
        chunks = fit_chunks(chunks, count_tokens, max_tokens)
    return chunks


def chunk_label(chunk: Chunk, payload_span: tuple[int, int] | None) -> int | None:
    """Training label of one chunk of a segment: 1 if it holds at least half of the injected
    payload (or is at least half payload), 0 if it holds none of it, None if it holds only a
    sliver (ambiguous, left out of training). Benign segments have no payload span."""
    if payload_span is None:
        return 0
    ps, pe = payload_span
    overlap = max(0, min(chunk.end, pe) - max(chunk.start, ps))
    if overlap == 0:
        return 0
    if overlap >= 0.5 * (pe - ps) or overlap >= 0.5 * (chunk.end - chunk.start):
        return 1
    return None
