"""Threshold calibration math, rates and intervals, AUROC, report slices."""

import math
import random

import pytest

from injbench.detect.calibration import (Calibration, auroc, flag, rate, sigmoid,
                                         threshold_at_fpr, wilson)
from injbench.detect.evaluate import metrics
from injbench.detect.latency import percentile


def fpr_at(scores, t):
    return sum(flag(scores, t)) / len(scores)


def test_threshold_allows_floor_of_target_times_n_false_positives():
    scores = [float(i) for i in range(100)]  # 0..99
    t = threshold_at_fpr(scores, 0.01)  # k = 1: only the top score may be flagged
    assert flag(scores, t).count(True) == 1 and 98 < t < 99.0001
    t = threshold_at_fpr(scores, 0.05)
    assert flag(scores, t).count(True) == 5


def test_small_sets_flag_nothing_benign():
    scores = [0.1, 0.5, 3.0]  # n < 100: k = 0
    t = threshold_at_fpr(scores, 0.01)
    assert t > 3.0 and not any(flag(scores, t))
    assert t == math.nextafter(3.0, math.inf)


def test_ties_at_the_boundary_are_not_flagged():
    scores = [5.0, 5.0, 5.0] + [0.0] * 97  # k = 1, but three tie at the top
    t = threshold_at_fpr(scores, 0.01)
    assert fpr_at(scores, t) == 0.0  # flagging any 5.0 would flag all three


def test_threshold_is_minimal_among_those_meeting_the_target():
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(1, 400)
        scores = [round(rng.gauss(0, 2), 1) for _ in range(n)]  # rounding makes ties
        target = rng.choice([0.0, 0.01, 0.05, 0.2])
        t = threshold_at_fpr(scores, target)
        assert fpr_at(scores, t) <= target + 1e-12
        lower = [s for s in set(scores) if s < t]
        if lower:  # the next candidate down already exceeds the target
            assert fpr_at(scores, max(lower)) > target


def test_threshold_input_validation():
    with pytest.raises(ValueError):
        threshold_at_fpr([], 0.01)
    with pytest.raises(ValueError):
        threshold_at_fpr([1.0], 1.0)
    with pytest.raises(ValueError):
        threshold_at_fpr([float("nan")], 0.01)


def test_wilson_interval():
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.7225, abs=1e-3)
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert all(math.isnan(x) for x in wilson(0, 0))
    r = rate([True, False, False, True]).to_dict()
    assert r["k"] == 2 and r["n"] == 4 and r["rate"] == 0.5 and r["ci95"][0] < 0.5 < r["ci95"][1]


def test_auroc():
    assert auroc([2, 3], [0, 1]) == 1.0
    assert auroc([0, 1], [2, 3]) == 0.0
    assert auroc([1, 1], [1, 1]) == 0.5
    assert auroc([1, 3], [2]) == 0.5
    assert auroc([2, 2, 3], [1, 2]) == pytest.approx((1 + 1 + 0.5 + 0.5 + 1 + 1) / 6)
    assert math.isnan(auroc([], [1]))


def test_sigmoid_is_stable():
    assert sigmoid(0) == 0.5 and sigmoid(1000) == 1.0 and sigmoid(-1000) == 0.0
    assert sigmoid(3) == pytest.approx(1 / (1 + math.exp(-3)))


def test_calibration_file_is_frozen(tmp_path):
    cal = Calibration(tag="t", detector="modernbert", model="modernbert-seed0", threshold=2.5,
                      target_fpr=0.01, val={"fpr": {"k": 0, "n": 10}})
    path = tmp_path / "t.json"
    cal.save(path)
    with pytest.raises(FileExistsError):
        cal.save(path)
    cal.save(path, overwrite=True)
    back = Calibration.load(path)
    assert back == cal and back.threshold_prob == pytest.approx(sigmoid(2.5))


def _row(label, **kw):
    base = {"label": label, "count": 1, "hard_negative": False, "hn_origin": None,
            "hn_kind": None, "seen_in_train": False, "field_kind": "body", "origin": "replay",
            "family": None, "goal": None, "goal_class": None, "family_seen": None,
            "templates": ["t"]}
    return {**base, **kw}


def test_metrics_slices():
    rows = [
        _row(0, hard_negative=True, hn_origin="world", hn_kind="escalation_request", count=3),
        _row(0, hard_negative=True, hn_origin="augment", hn_kind="correction"),
        _row(0, seen_in_train=True),
        _row(1, family="override", goal="g1", goal_class="a", family_seen=True),
        _row(1, family="override", goal="g2", goal_class="b", family_seen=True),
        _row(1, family="business_note", goal="g1", goal_class="a", family_seen=False),
    ]
    scores = [3.0, -1.0, -2.0, 4.0, -3.0, 1.0]
    m = metrics(rows, scores, threshold=0.0)
    assert m["fpr"]["k"] == 1 and m["fpr"]["n"] == 3
    assert m["fpr_occurrence_weighted"] == {"k": 3, "n": 5, "rate": 0.6}
    assert m["fpr_world_hard_negatives"]["k"] == 1
    assert m["fpr_augmented_hard_negatives"]["k"] == 0
    assert m["fpr_text_unseen_in_train"]["n"] == 2
    assert m["recall"]["k"] == 2 and m["recall"]["n"] == 3
    assert m["recall_seen_families"]["k"] == 1 and m["recall_seen_families"]["n"] == 2
    assert m["recall_unseen_families"]["k"] == 1
    assert m["recall_by_goal"]["g1"]["k"] == 2
    assert m["auroc_unseen_families"] == pytest.approx(2 / 3)


def test_percentile_matches_linear_interpolation():
    xs = [1, 2, 3, 4]
    assert percentile(xs, 50) == 2.5 and percentile(xs, 0) == 1 and percentile(xs, 100) == 4
    assert percentile([5], 95) == 5 and math.isnan(percentile([], 50))
    assert percentile(list(range(101)), 95) == 95
