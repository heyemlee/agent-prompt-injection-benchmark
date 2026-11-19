"""Tool-output segments and the fixed on-detect action (redaction).

A *segment* is one free-text string value inside a tool result, which `Environment.call` hands
to the output filter as dicts and lists before JSON encoding: a ticket title, a note body, a KB
article body, a free-text `run_sql` cell. Dict keys are never segments, and non-string values
(numbers, booleans, None) are never touched.

The dataset builder and the defense arms both use `iter_segments`, so the detectors are trained
and calibrated on exactly the units they scan at run time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

# The on-detect action is identical across detector arms: each flagged segment is replaced by
# this one fixed notice, so arms differ only in the detector.
REDACTION_NOTICE = "[redacted: this text was flagged as a possible prompt injection]"

# Strings without whitespace (keys, dates, emails, enum values) cannot carry a natural-language
# instruction and are not scanned, unless they are long enough to hide one.
LONG_TOKEN_CHARS = 64

SegmentPath = tuple[Any, ...]
SegmentScorer = Callable[[list[str]], Sequence[float]]


def is_scannable(text: str) -> bool:
    t = text.strip()
    return bool(t) and (len(t) >= LONG_TOKEN_CHARS or any(c.isspace() for c in t))


def iter_segments(obj: Any, path: SegmentPath = ()) -> Iterator[tuple[SegmentPath, str]]:
    """Yields (path, text) for every scannable string value, depth first, in document order.

    A path holds dict keys and list indices, e.g. ("notes", 1, "body").
    """
    if isinstance(obj, str):
        if is_scannable(obj):
            yield path, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from iter_segments(value, path + (key,))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            yield from iter_segments(value, path + (i,))


def replace_segments(obj: Any, paths: set[SegmentPath], notice: str = REDACTION_NOTICE,
                     path: SegmentPath = ()) -> Any:
    """A copy of `obj` with the strings at `paths` replaced by `notice`. The input is not
    modified; keys and every other value are kept as they are."""
    if isinstance(obj, str):
        return notice if path in paths else obj
    if isinstance(obj, dict):
        return {k: replace_segments(v, paths, notice, path + (k,)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_segments(v, paths, notice, path + (i,)) for i, v in enumerate(obj)]
    if isinstance(obj, tuple):
        return tuple(replace_segments(v, paths, notice, path + (i,)) for i, v in enumerate(obj))
    return obj


@dataclass
class FilterStats:
    """Counters a redaction filter keeps, for latency and redaction-rate reporting."""

    calls: int = 0
    segments: int = 0
    flagged: int = 0
    filtered_calls: int = 0
    seconds: float = 0.0
    per_call_seconds: list[float] = field(default_factory=list)

    def add(self, other: "FilterStats") -> None:
        self.calls += other.calls
        self.segments += other.segments
        self.flagged += other.flagged
        self.filtered_calls += other.filtered_calls
        self.seconds += other.seconds
        self.per_call_seconds += other.per_call_seconds


def make_redaction_filter(score: SegmentScorer, threshold: float, *,
                          notice: str = REDACTION_NOTICE, stats: FilterStats | None = None):
    """The output filter of a detector arm: `fn(tool_name, result) -> (new_result, filtered)`.

    `score` maps segment texts to detector scores (higher means more likely injected); a segment
    is flagged when its score is at least `threshold`. All segments of one tool result are
    scored in one call so the detector can batch them.
    """
    stats = stats if stats is not None else FilterStats()

    def output_filter(tool_name: str, result: Any) -> tuple[Any, bool]:
        t0 = time.perf_counter()
        segments = list(iter_segments(result))
        flagged: set[SegmentPath] = set()
        if segments:
            scores = list(score([text for _, text in segments]))
            if len(scores) != len(segments):
                raise RuntimeError(f"detector returned {len(scores)} scores for "
                                   f"{len(segments)} segments")
            flagged = {p for (p, _), s in zip(segments, scores) if s >= threshold}
        new = replace_segments(result, flagged, notice) if flagged else result
        dt = time.perf_counter() - t0
        stats.calls += 1
        stats.segments += len(segments)
        stats.flagged += len(flagged)
        stats.filtered_calls += bool(flagged)
        stats.seconds += dt
        stats.per_call_seconds.append(dt)
        return new, bool(flagged)

    output_filter.stats = stats  # type: ignore[attr-defined]
    return output_filter
