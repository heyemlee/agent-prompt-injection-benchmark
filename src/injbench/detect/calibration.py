"""Thresholds at a target false-positive rate, rates with confidence intervals, AUROC.

Scores are log-odds of "injected" (higher means more likely injected). Log-odds do not saturate
the way float32 probabilities do near 1.0, so thresholds between confident scores stay distinct.
A segment is flagged when its score is at least the threshold.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


def threshold_at_fpr(benign_scores: Sequence[float], target_fpr: float) -> float:
    """The smallest threshold whose false-positive rate on `benign_scores` is <= target_fpr.

    With n benign scores, at most k = floor(target_fpr * n) of them may be flagged. The
    threshold sits just above the (k+1)-th largest score, so ties at that score are not flagged
    either. Recall is maximal among thresholds that meet the target. With n < 1/target_fpr, k is
    0 and the threshold sits just above the largest benign score.
    """
    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be in [0, 1)")
    scores = sorted((float(s) for s in benign_scores), reverse=True)
    if not scores:
        raise ValueError("need at least one benign score to calibrate")
    if any(math.isnan(s) for s in scores):
        raise ValueError("benign scores contain NaN")
    k = math.floor(target_fpr * len(scores) + 1e-9)
    return math.nextafter(scores[k], math.inf)


def flag(scores: Iterable[float], threshold: float) -> list[bool]:
    return [s >= threshold for s in scores]


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, center - half)  # exact at the ends (no rounding error)
    hi = 1.0 if k == n else min(1.0, center + half)
    return (lo, hi)


@dataclass
class Rate:
    k: int
    n: int

    @property
    def value(self) -> float:
        return self.k / self.n if self.n else math.nan

    def to_dict(self) -> dict:
        lo, hi = wilson(self.k, self.n)
        return {"k": self.k, "n": self.n, "rate": self.value, "ci95": [lo, hi]}


def rate(flags: Iterable[bool]) -> Rate:
    flags = list(flags)
    return Rate(sum(bool(f) for f in flags), len(flags))


def auroc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Probability that a random injected segment outscores a random benign one (ties count
    half): the Mann-Whitney U statistic, computed by ranking."""
    if not pos or not neg:
        return math.nan
    allv = sorted([(float(s), 1) for s in pos] + [(float(s), 0) for s in neg])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2  # ranks i+1 .. j share their average
        rank_sum += avg_rank * sum(lbl for _, lbl in allv[i:j])
        i = j
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@dataclass
class Calibration:
    """A frozen threshold for one detector, chosen on validation data."""

    tag: str  # file name, also what an arm refers to
    detector: str  # key in detectors.DETECTOR_SPECS
    model: str  # hub id or checkpoint directory the scores came from
    threshold: float  # log-odds; flag when score >= threshold
    target_fpr: float
    val: dict = field(default_factory=dict)  # achieved FPR and recall on validation
    dataset_fingerprint: str = ""
    chunking: dict = field(default_factory=dict)
    device: str = ""
    created: str = ""
    injbench_version: str = ""

    @property
    def threshold_prob(self) -> float:
        return sigmoid(self.threshold)

    def to_dict(self) -> dict:
        return {**asdict(self), "threshold_prob": self.threshold_prob}

    def save(self, path: str | Path, overwrite: bool = False) -> Path:
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; thresholds are frozen once chosen "
                                  f"(pass --force to recalibrate on purpose)")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        raw = json.loads(Path(path).read_text())
        raw.pop("threshold_prob", None)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})
