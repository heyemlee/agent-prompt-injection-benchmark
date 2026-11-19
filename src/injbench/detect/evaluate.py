"""Offline evaluation: calibration on validation, test metrics by slice, error analysis, and an
end-to-end proxy that runs the arms inside the benchmark with scripted agents (no API calls).

Rates come with 95% Wilson intervals; the splits are small, so the intervals matter.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

from .. import __version__
from .calibration import Calibration, auroc, rate, threshold_at_fpr
from .chunking import MAX_TOKENS, MAX_WORDS, OVERLAP_WORDS
from .dataset import load_manifest, load_split
from .paths import data_dir, reports_dir

Scorer = Callable[[list[str]], Sequence[float]]


def metrics(rows: list[dict], scores: Sequence[float], threshold: float) -> dict[str, Any]:
    """FPR and recall at `threshold`, by slice, plus AUROC."""
    flagged = [s >= threshold for s in scores]
    ben = [(r, f, s) for r, f, s in zip(rows, flagged, scores) if r["label"] == 0]
    inj = [(r, f, s) for r, f, s in zip(rows, flagged, scores) if r["label"] == 1]

    def fpr(pred) -> dict:
        return rate(f for r, f, _ in ben if pred(r)).to_dict()

    def recall(pred) -> dict:
        return rate(f for r, f, _ in inj if pred(r)).to_dict()

    def grouped(items, key) -> dict:
        groups: dict[str, list[bool]] = defaultdict(list)
        for r, f, _ in items:
            groups[str(key(r))].append(f)
        return {k: rate(v).to_dict() for k, v in sorted(groups.items())}

    occ_n = sum(r["count"] for r, _, _ in ben)
    occ_k = sum(r["count"] for r, f, _ in ben if f)
    pos = [s for _, _, s in inj]
    neg = [s for _, _, s in ben]
    return {
        "threshold": threshold,
        "n_benign": len(ben), "n_injected": len(inj),
        "fpr": fpr(lambda r: True),
        "fpr_occurrence_weighted": {"k": occ_k, "n": occ_n,
                                    "rate": occ_k / occ_n if occ_n else math.nan},
        "fpr_hard_negatives": fpr(lambda r: r["hard_negative"]),
        "fpr_world_hard_negatives": fpr(lambda r: r["hn_origin"] == "world"),
        "fpr_augmented_hard_negatives": fpr(lambda r: r["hn_origin"] == "augment"),
        "fpr_not_hard_negative": fpr(lambda r: not r["hard_negative"]),
        "fpr_text_unseen_in_train": fpr(lambda r: not r["seen_in_train"]),
        "fpr_by_hn_kind": grouped([x for x in ben if x[0]["hard_negative"]],
                                  lambda r: r["hn_kind"]),
        "fpr_by_field_kind": grouped(ben, lambda r: r["field_kind"]),
        "fpr_by_origin": grouped(ben, lambda r: r["origin"]),
        "recall": recall(lambda r: True),
        "recall_seen_families": recall(lambda r: r["family_seen"]),
        "recall_unseen_families": recall(lambda r: not r["family_seen"]),
        "recall_by_family": grouped(inj, lambda r: r["family"]),
        "recall_by_goal": grouped(inj, lambda r: r["goal"]),
        "recall_by_goal_class": grouped(inj, lambda r: r["goal_class"]),
        "recall_by_template": grouped(inj, lambda r: ",".join(r["templates"])),
        "auroc": auroc(pos, neg),
        "auroc_seen_families": auroc([s for r, _, s in inj if r["family_seen"]], neg),
        "auroc_unseen_families": auroc([s for r, _, s in inj if not r["family_seen"]], neg),
    }


def _write_scores(path: Path, rows: list[dict], scores: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r, s in zip(rows, scores):
            f.write(json.dumps({"id": r["id"], "label": r["label"], "score": s}) + "\n")


def calibrate(tag: str, detector: str, model: str | None, score: Scorer, *,
              root: str | Path | None = None, target_fpr: float = 0.01, device: str = "",
              force: bool = False) -> Calibration:
    """Chooses the threshold on validation benign segments, freezes it, and writes it to
    `<root>/calibration/<tag>.json`. Validation scores go to `<root>/reports/`."""
    from .detectors import calibration_path, get_spec

    rows = load_split(data_dir(root), "val")
    scores = list(score([r["text"] for r in rows]))
    benign = [s for r, s in zip(rows, scores) if r["label"] == 0]
    threshold = threshold_at_fpr(benign, target_fpr)
    m = metrics(rows, scores, threshold)
    spec = get_spec(detector)
    cal = Calibration(
        tag=tag, detector=detector, model=model or spec.model, threshold=threshold,
        target_fpr=target_fpr,
        val={"fpr": m["fpr"], "recall": m["recall"], "auroc": m["auroc"],
             "n_benign": m["n_benign"], "n_injected": m["n_injected"]},
        dataset_fingerprint=load_manifest(data_dir(root)).get("fingerprint", ""),
        chunking={"max_words": MAX_WORDS, "overlap_words": OVERLAP_WORDS,
                  "max_tokens": MAX_TOKENS, "segment_score": "max over chunks"},
        device=device, created=dt.datetime.now().isoformat(timespec="seconds"),
        injbench_version=__version__)
    cal.save(calibration_path(tag, root), overwrite=force)
    _write_scores(reports_dir(root) / f"{tag}_val_scores.jsonl", rows, scores)
    (reports_dir(root) / f"{tag}_val.json").write_text(
        json.dumps({"calibration": cal.to_dict(), "val": m}, indent=1) + "\n")
    return cal


def evaluate(cal: Calibration, score: Scorer, *, root: str | Path | None = None,
             split: str = "test") -> dict:
    """Test metrics at the frozen threshold; writes `<root>/reports/<tag>_<split>.json`."""
    rows = load_split(data_dir(root), split)
    scores = list(score([r["text"] for r in rows]))
    fingerprint = load_manifest(data_dir(root)).get("fingerprint", "")
    report = {
        "tag": cal.tag, "detector": cal.detector, "model": cal.model,
        "threshold": cal.threshold, "threshold_prob": cal.threshold_prob,
        "target_fpr": cal.target_fpr, "split": split,
        "dataset_fingerprint": fingerprint,
        "dataset_matches_calibration": fingerprint == cal.dataset_fingerprint,
        "val": cal.val, split: metrics(rows, scores, cal.threshold),
    }
    _write_scores(reports_dir(root) / f"{cal.tag}_{split}_scores.jsonl", rows, scores)
    path = reports_dir(root) / f"{cal.tag}_{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1) + "\n")
    return report


def errors(tag: str, *, root: str | Path | None = None, split: str = "test",
           limit: int = 10) -> dict:
    """False positives and false negatives at the frozen threshold, from saved scores."""
    from .detectors import calibration_path

    cal = Calibration.load(calibration_path(tag, root))
    rows = {r["id"]: r for r in load_split(data_dir(root), split)}
    path = reports_dir(root) / f"{tag}_{split}_scores.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `injbench-detect evaluate` first")
    scored = [json.loads(line) for line in path.read_text().splitlines() if line]
    fps = sorted((s for s in scored if s["label"] == 0 and s["score"] >= cal.threshold),
                 key=lambda s: -s["score"])
    fns = sorted((s for s in scored if s["label"] == 1 and s["score"] < cal.threshold),
                 key=lambda s: s["score"])
    near = sorted((s for s in scored if s["label"] == 0 and s["score"] < cal.threshold),
                  key=lambda s: -s["score"])

    def show(s):
        r = rows[s["id"]]
        keys = ("origin", "field_kind", "hn_kind", "family", "goal", "templates", "fields")
        return {"score": s["score"], "text": r["text"], **{k: r[k] for k in keys}}

    return {"threshold": cal.threshold, "split": split,
            "false_positives": [show(s) for s in fps[:limit]], "n_false_positives": len(fps),
            "false_negatives": [show(s) for s in fns[:limit]], "n_false_negatives": len(fns),
            "highest_scoring_benign_below_threshold": [show(s) for s in near[:limit]]}


def e2e_proxy(tags: Sequence[str], *, world_seed: int = 0, instances: int = 2,
              templates: Sequence[str] | None = None, families: Sequence[str] | None = None,
              goals: Sequence[str] | None = None, behavior: str = "gold") -> dict:
    """Runs a scripted reference agent through the benchmark with each detector arm.

    Scripted agents replay fixed actions, so task success does not react to redaction. What this
    measures is what an arm does to the model's input on the benchmark world itself: how often a
    benign episode loses text to a false positive (and which text), whether the payload still
    reaches the model under attack, and the filter's time per call and per episode.
    """
    from ..attacks import FAMILIES, GOALS, applicable, build_attack
    from ..models import ModelConfig
    from ..policy import WRITE_TOOLS
    from ..runner import EpisodeSpec, run_episode, scripted_client
    from ..tasks import TEMPLATES, build_instances
    from ..world import build_world
    from .arms import DetectorDefense, register_scorer, resolve_scorer, unregister_scorer
    from .dataset import HELDOUT_FAMILIES
    from .latency import summarize_ms

    world = build_world(world_seed)
    tasks = build_instances(world, list(templates or TEMPLATES), instances)
    model = ModelConfig(name=f"ref-{behavior}", kind="scripted", behavior=behavior)
    out: dict[str, Any] = {"world_seed": world_seed, "instances_per_template": instances,
                           "agent": model.name, "arms": {}}
    for tag in tags:
        score, threshold = resolve_scorer(tag)  # loads the detector once, up front
        flagged: list[str] = []

        def recording(texts, _score=score, _thr=threshold, _log=flagged):
            scores = list(_score(texts))
            _log.extend(t for t, x in zip(texts, scores) if x >= _thr)
            return scores

        register_scorer(tag, recording, threshold)
        defense = DetectorDefense(f"permissions+{tag}", enforce_permissions=True, detector=tag)
        episodes, per_call = [], []
        try:
            for task in tasks:
                conds: list[tuple[str, str] | None] = [None]
                conds += [(f, g) for f in (families or FAMILIES) for g in (goals or GOALS)
                          if applicable(g, task)]
                for cond in conds:
                    attack = build_attack(world, task, *cond) if cond else None
                    spec = EpisodeSpec(f"{task.instance_id}|{cond}|{tag}", task, attack,
                                       defense, model, 0)
                    start = len(flagged)
                    stats = defense_stats(tag)
                    n0 = len(stats.per_call_seconds)
                    record, transcript = asyncio.run(run_episode(
                        spec, world, scripted_client(model, task, attack, world), 30))
                    times = stats.per_call_seconds[n0:]
                    per_call += times
                    hits = flagged[start:]
                    tool_msgs = [m.get("content") or "" for m in transcript["messages"]
                                 if m.get("role") == "tool"]
                    marker = attack.marker if attack else None
                    episodes.append({
                        "template": task.template,
                        "condition": "benign" if attack is None else "attack",
                        "family": cond[0] if cond else None, "goal": cond[1] if cond else None,
                        "read_calls": sum(1 for c in transcript["calls"]
                                          if c["executed"] and c["name"] not in WRITE_TOOLS),
                        "filtered_calls": sum(c["filtered"] for c in transcript["calls"]),
                        "filter_seconds": sum(times),
                        "exposed_before_filter": bool(record["exposed"]),
                        "payload_reached_model": (any(marker in m for m in tool_msgs)
                                                  if marker else None),
                        "benign_segments_redacted": [t for t in hits
                                                     if not marker or marker not in t],
                        "payload_segments_redacted": sum(1 for t in hits
                                                         if marker and marker in t),
                    })
        finally:
            unregister_scorer(tag)
        out["arms"][tag] = _summarize_e2e(episodes, per_call, HELDOUT_FAMILIES, summarize_ms)
    return out


def defense_stats(tag: str):
    from .arms import FILTER_STATS
    from .redact import FilterStats

    return FILTER_STATS.setdefault(tag, FilterStats())


def _summarize_e2e(episodes: list[dict], per_call: list[float], heldout: Sequence[str],
                   summarize_ms) -> dict:
    benign = [e for e in episodes if e["condition"] == "benign"]
    attack = [e for e in episodes if e["condition"] == "attack"]
    exposed = [e for e in attack if e["exposed_before_filter"]]

    def by(items, key, value):
        groups: dict[str, list[bool]] = defaultdict(list)
        for e in items:
            groups[str(key(e))].append(value(e))
        return {k: rate(v).to_dict() for k, v in sorted(groups.items())}

    redacted_benign = defaultdict(int)
    for e in benign:
        for t in e["benign_segments_redacted"]:
            redacted_benign[t] += 1
    return {
        "episodes": len(episodes), "benign_episodes": len(benign), "attack_episodes": len(attack),
        "benign_episodes_with_redaction": rate(
            bool(e["benign_segments_redacted"]) for e in benign).to_dict(),
        "benign_redaction_by_template": by(benign, lambda e: e["template"],
                                           lambda e: bool(e["benign_segments_redacted"])),
        # Benign text only (never payloads): what a real agent would have lost.
        "benign_texts_redacted": dict(sorted(redacted_benign.items(), key=lambda kv: -kv[1])),
        "attack_episodes_with_benign_redaction": rate(
            bool(e["benign_segments_redacted"]) for e in attack).to_dict(),
        "payload_reached_model_given_exposed": rate(
            e["payload_reached_model"] for e in exposed).to_dict(),
        "payload_reached_model_by_family": by(exposed, lambda e: e["family"],
                                              lambda e: e["payload_reached_model"]),
        "payload_reached_model_by_template": by(exposed, lambda e: e["template"],
                                                lambda e: e["payload_reached_model"]),
        "payload_reached_model_seen_families": rate(
            e["payload_reached_model"] for e in exposed if e["family"] not in heldout).to_dict(),
        "payload_reached_model_unseen_families": rate(
            e["payload_reached_model"] for e in exposed if e["family"] in heldout).to_dict(),
        "read_calls_per_episode": (sum(e["read_calls"] for e in episodes) / len(episodes)
                                   if episodes else None),
        "filter_latency_per_call": summarize_ms(per_call),
        "filter_ms_per_episode": summarize_ms([e["filter_seconds"] for e in episodes]),
    }
