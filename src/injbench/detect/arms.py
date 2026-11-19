"""Detector defense arms: least-privilege permissions plus an injection detector.

Each arm's output filter scans every segment of every executed read tool's result and replaces
each flagged segment with one fixed notice (`redact.REDACTION_NOTICE`). The on-detect action is
identical across arms, so they differ only in the detector and its frozen threshold.

Weights load lazily, when an arm first builds its filter (at the start of an episode, before any
model call), and are shared by every episode in the process. Importing this module never imports
torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..defenses import Defense, OutputFilter
from .redact import FilterStats, SegmentScorer, make_redaction_filter

# arm name -> calibration tag (the file `<root>/calibration/<tag>.json` names the detector,
# its weights and its threshold)
ARM_DETECTORS: dict[str, str] = {
    "permissions+promptguard2": "promptguard2",
    "permissions+modernbert": "modernbert",
}

# Filter counters per calibration tag, accumulated over every episode in this process.
FILTER_STATS: dict[str, FilterStats] = {}

_OVERRIDES: dict[str, tuple[SegmentScorer, float]] = {}


def register_scorer(tag: str, score: SegmentScorer, threshold: float) -> None:
    """Makes arms with this calibration tag use `score` and `threshold` instead of loading a
    model (tests use fake detectors this way)."""
    _OVERRIDES[tag] = (score, threshold)


def unregister_scorer(tag: str) -> None:
    _OVERRIDES.pop(tag, None)


def resolve_scorer(tag: str, root: str | Path | None = None,
                   device: str | None = None) -> tuple[SegmentScorer, float]:
    if tag in _OVERRIDES:
        return _OVERRIDES[tag]
    from .detectors import load_calibrated

    detector, cal = load_calibrated(tag, root, device)
    return detector.score_segments, cal.threshold


@dataclass(frozen=True)
class DetectorDefense(Defense):
    detector: str = ""  # calibration tag

    def make_output_filter(self) -> OutputFilter:
        score, threshold = resolve_scorer(self.detector)
        stats = FILTER_STATS.setdefault(self.detector, FilterStats())
        return make_redaction_filter(score, threshold, stats=stats)


DETECTOR_ARMS: dict[str, DetectorDefense] = {
    name: DetectorDefense(name, enforce_permissions=True, detector=tag)
    for name, tag in ARM_DETECTORS.items()
}


def arm_names() -> Sequence[str]:
    return tuple(DETECTOR_ARMS)
