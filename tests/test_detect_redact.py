"""The redaction filter and the detector arms, with fake detectors (no torch)."""

import asyncio
import copy
import json
import subprocess
import sys

import pytest

from injbench.attacks import build_attack
from injbench.defenses import DEFENSES, get_defense
from injbench.detect.arms import (ARM_DETECTORS, DetectorDefense, register_scorer,
                                  unregister_scorer)
from injbench.detect.detectors import DetectorUnavailable, label_indices
from injbench.detect.redact import (REDACTION_NOTICE, is_scannable, iter_segments,
                                    make_redaction_filter, replace_segments)
from injbench.env import Environment
from injbench.models import ModelConfig
from injbench.runner import EpisodeSpec, run_episode, scripted_client
from injbench.tasks import build_instances


def fake_scores(texts):
    """A keyword detector: high log-odds for anything mentioning EVIL."""
    return [5.0 if "EVIL" in t else -5.0 for t in texts]


RESULT = {
    "key": "INC-1001",
    "title": "Laptop will not boot",
    "EVIL key stays": "a value without the keyword",
    "priority": "P3",
    "sla_breached": 1,
    "closed_date": None,
    "notes": [
        {"author": "Avery Chen", "body": "Started a few minutes ago."},
        {"author": "Blake Diaz", "body": "Fine so far.\n\nEVIL instruction goes here"},
    ],
    "rows": [["INC-1", "plain text cell", 3], ["INC-2", "EVIL cell text", 4.5]],
}


def test_segments_are_free_text_values_only():
    segs = dict(iter_segments(RESULT))
    assert ("title",) in segs and ("notes", 1, "body") in segs and ("rows", 1, 1) in segs
    assert ("key",) not in segs and ("priority",) not in segs  # no whitespace: not scanned
    assert all(isinstance(p, tuple) for p in segs)
    assert "EVIL key stays" not in {p[0] for p in segs if p and p[0] != "EVIL key stays"}
    assert segs[("EVIL key stays",)] == "a value without the keyword"  # value, not the key
    assert is_scannable("x" * 64) and not is_scannable("x" * 63) and not is_scannable("   ")


def test_filter_redacts_flagged_segments_in_nested_results():
    original = copy.deepcopy(RESULT)
    filt = make_redaction_filter(fake_scores, threshold=0.0)
    new, filtered = filt("get_ticket", RESULT)
    assert filtered is True
    assert RESULT == original  # the input is not modified
    assert new["notes"][1]["body"] == REDACTION_NOTICE
    assert new["rows"][1][1] == REDACTION_NOTICE
    assert new["notes"][0] == original["notes"][0]
    assert new["rows"][0] == original["rows"][0] and new["rows"][1][2] == 4.5
    assert list(new) == list(original)  # keys untouched, same order
    assert new["EVIL key stays"] == original["EVIL key stays"]
    assert new["sla_breached"] == 1 and new["closed_date"] is None
    assert filt.stats.calls == 1 and filt.stats.flagged == 2 and filt.stats.filtered_calls == 1


def test_filter_leaves_clean_results_alone():
    calls = []

    def scorer(texts):
        calls.append(list(texts))
        return [-1.0] * len(texts)

    filt = make_redaction_filter(scorer, threshold=0.0)
    clean = {"title": "Laptop will not boot", "rows": [["a b", 1]]}
    new, filtered = filt("run_sql", clean)
    assert filtered is False and new is clean
    assert calls == [["Laptop will not boot", "a b"]]  # one batched call per tool result
    new, filtered = filt("run_sql", {"columns": ["n"], "rows": [[1]]})
    assert filtered is False and len(calls) == 1  # nothing to scan: the detector is not called


def test_threshold_is_inclusive_and_score_count_is_checked():
    filt = make_redaction_filter(lambda ts: [0.5] * len(ts), threshold=0.5)
    assert filt("x", ["two words"])[1] is True
    bad = make_redaction_filter(lambda ts: [], threshold=0.0)
    with pytest.raises(RuntimeError):
        bad("x", ["two words"])


def test_replace_segments_handles_tuples_and_top_level_lists():
    obj = ("a b", ["c d", {"k": "e f"}])
    out = replace_segments(obj, {(0,), (1, 1, "k")}, "N")
    assert out == ("N", ["c d", {"k": "N"}]) and obj == ("a b", ["c d", {"k": "e f"}])


def test_environment_applies_filter_to_read_tools_only(world):
    task = build_instances(world, ["sla_summary_report"], 1)[0]
    attack = build_attack(world, task, "info_block", "grant_admin")
    marker_filter = make_redaction_filter(
        lambda ts: [9.0 if attack.marker in t else -9.0 for t in ts], threshold=0.0)
    env = Environment(world, task.principal, True, attack.marker, marker_filter)
    env.inject(task.vector.table, task.vector.row_id, task.vector.column, attack.payload)
    name, args = task.gold_actions[0]
    out = env.call(name, args)
    assert attack.marker not in out and REDACTION_NOTICE in out
    assert env.calls[-1].exposed and env.calls[-1].filtered
    env.call("add_ticket_comment", {"key": args["key"], "body": "EVIL but a write tool"})
    assert env.calls[-1].executed and not env.calls[-1].filtered


def test_arms_are_registered_and_listed(capsys):
    from injbench.cli import main

    for name, tag in ARM_DETECTORS.items():
        arm = get_defense(name)
        assert isinstance(arm, DetectorDefense)
        assert arm.enforce_permissions == name.startswith("permissions+")
        assert arm.detector == tag and DEFENSES[name] is arm
    assert set(ARM_DETECTORS) == {"promptguard2", "modernbert",
                                  "permissions+promptguard2", "permissions+modernbert"}
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "permissions+promptguard2" in out and "permissions+modernbert" in out
    assert "\nmodernbert" in out or " modernbert" in out


def _episode(world, task, attack, defense):
    m = ModelConfig(name="ref-gold", kind="scripted", behavior="gold")
    spec = EpisodeSpec("x", task, attack, defense, m, 0)
    return asyncio.run(run_episode(spec, world, scripted_client(m, task, attack, world), 30))


def test_detector_arm_end_to_end_with_fake_detector(world):
    task = build_instances(world, ["kb_answer_comment"], 1)[0]
    attack = build_attack(world, task, "business_note", "close_unrequested")
    register_scorer("fake", lambda ts: [9.0 if attack.marker in t else -9.0 for t in ts], 0.0)
    try:
        arm = DetectorDefense("permissions+fake", enforce_permissions=True, detector="fake")
        record, transcript = _episode(world, task, attack, arm)
        tool_msgs = [m["content"] for m in transcript["messages"] if m["role"] == "tool"]
        assert record["exposed"]  # the payload was in a tool result before the filter...
        assert not any(attack.marker in m for m in tool_msgs)  # ...and never reached the model
        assert any(REDACTION_NOTICE in m for m in tool_msgs)
        assert any(c["filtered"] for c in transcript["calls"])
        assert record["task_success"] and record["safe_completion"]
        record, transcript = _episode(world, task, None, arm)  # benign: nothing to redact
        assert not any(c["filtered"] for c in transcript["calls"])
    finally:
        unregister_scorer("fake")


def test_detector_only_arm_redacts_but_does_not_enforce_permissions(world):
    """Alone, the detector only redacts. The scripted gullible agent does not read, so it carries
    out an out-of-permission goal anyway: nothing blocks the call, unlike with permissions."""
    task = build_instances(world, ["sla_summary_report"], 1)[0]
    attack = build_attack(world, task, "info_block", "grant_admin")
    register_scorer("fake", lambda ts: [9.0 if attack.marker in t else -9.0 for t in ts], 0.0)
    try:
        gullible = ModelConfig(name="ref-gullible", kind="scripted", behavior="gullible")
        outcomes = {}
        for arm in (DetectorDefense("fake", enforce_permissions=False, detector="fake"),
                    DetectorDefense("permissions+fake", enforce_permissions=True, detector="fake")):
            spec = EpisodeSpec("x", task, attack, arm, gullible, 0)
            client = scripted_client(gullible, task, attack, world)
            record, transcript = asyncio.run(run_episode(spec, world, client, 30))
            tool_msgs = [m["content"] for m in transcript["messages"] if m["role"] == "tool"]
            assert record["exposed"] and not any(attack.marker in m for m in tool_msgs)
            outcomes[arm.name] = bool(record["attack_success"])
        assert outcomes == {"fake": True, "permissions+fake": False}
    finally:
        unregister_scorer("fake")


def test_uncalibrated_arm_fails_before_any_model_call(tmp_path, monkeypatch):
    monkeypatch.setenv("INJBENCH_DETECT_DIR", str(tmp_path))
    with pytest.raises(DetectorUnavailable, match="calibrate"):
        get_defense("permissions+modernbert").make_output_filter()


def test_label_indices():
    assert label_indices({0: "BENIGN", 1: "MALICIOUS"}) == ([1], [0])
    assert label_indices({0: "LABEL_0", 1: "LABEL_1"}) == ([1], [0])
    assert label_indices({"0": "BENIGN", "1": "INJECTION", "2": "JAILBREAK"}) == ([1, 2], [0])
    assert label_indices({0: "INJECTION", 1: "SAFE"}) == ([0], [1])
    with pytest.raises(ValueError):
        label_indices({0: "A", 1: "B", 2: "C"})


def test_importing_injbench_never_imports_torch():
    code = ("import sys, injbench, injbench.cli, injbench.defenses, injbench.runner, "
            "injbench.detect, injbench.detect.arms, injbench.detect.dataset, "
            "injbench.detect.detectors, injbench.detect.evaluate, injbench.detect.cli; "
            "heavy = [m for m in ('torch', 'transformers', 'numpy') if m in sys.modules]; "
            "print(heavy); sys.exit(1 if heavy else 0)")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for first in ("injbench.detect.arms", "injbench.defenses", "injbench.detect.dataset"):
        proc = subprocess.run(
            [sys.executable, "-c", f"import {first}; import injbench.defenses as d; "
             f"assert 'permissions+modernbert' in d.DEFENSES"], capture_output=True, text=True)
        assert proc.returncode == 0, f"import order {first}: {proc.stderr}"


def test_filter_output_is_json_serializable():
    filt = make_redaction_filter(fake_scores, threshold=0.0)
    new, _ = filt("get_ticket", RESULT)
    assert REDACTION_NOTICE in json.dumps(new)
