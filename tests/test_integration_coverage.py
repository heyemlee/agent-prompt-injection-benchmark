"""Cross-stream checks added when M2 (templates, spotlighting), M3 (detectors), M4 (attacks) and
M6 (Inspect) were merged: every template's injection vector is covered by each defense that
inspects tool output, not only the four M1 templates the defenses were written against."""

import json

import pytest

from injbench.attacks import FAMILIES, GOALS, applicable, build_attack
from injbench.defenses import get_defense
from injbench.detect.dataset import DatasetConfig, build_dataset
from injbench.detect.redact import REDACTION_NOTICE, make_redaction_filter
from injbench.env import Environment
from injbench.tasks import TEMPLATES, build_instances
from injbench.world import build_world


@pytest.fixture(scope="module")
def world():
    return build_world(0)


def _cases(world):
    """One instance per template, attacked by every family with its first applicable goal."""
    for template in TEMPLATES:
        task = build_instances(world, [template], 1)[0]
        goal = next(g for g in GOALS if applicable(g, task))
        for family in FAMILIES:
            yield task, build_attack(world, task, family, goal)


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _strings(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _strings(value)


def _gold_reads(world, task, attack, output_filter=None, renderer=None):
    """Run the gold actions; yield (tool name, call record, decoded tool output) per call."""
    env = Environment(world, task.principal, False, attack.marker, output_filter,
                      output_renderer=renderer)
    env.inject(task.vector.table, task.vector.row_id, task.vector.column, attack.payload)
    for name, args in task.gold_actions:
        content = env.call(name, args)
        yield name, env.calls[-1], list(_strings(json.loads(content)))


@pytest.mark.parametrize("arm", ["spotlight_delimit", "spotlight_datamark", "spotlight_base64"])
def test_spotlighting_marks_every_template_payload(world, arm):
    for task, attack in _cases(world):
        renderer = get_defense(arm).for_episode(f"{task.instance_id}|a0").make_output_renderer()
        exposed_reads = 0
        for name, rec, texts in _gold_reads(world, task, attack, renderer=renderer):
            if not rec.exposed:
                continue
            exposed_reads += 1
            where = f"{task.template} / {attack.family} / {name}"
            carrying = [t for t in texts if attack.marker in renderer.unmark(t)]
            assert carrying, f"payload not recoverable after rendering: {where}"
            for text in carrying:
                if renderer.mode == "delimit":
                    assert text.startswith(renderer.open_tag), where
                    assert text.endswith(renderer.close_tag), where
                else:
                    assert attack.marker not in text, f"payload left unmarked: {where}"
        assert exposed_reads, f"gold path never reads the payload: {task.template}"


def test_oracle_detector_keeps_every_template_payload_from_the_model(world):
    """With a detector that flags exactly the payload-bearing segments, redaction removes the
    payload from every read on the gold path, so the detector arms' segment walker covers all
    templates' untrusted fields (ticket titles and notes, KB bodies, report bodies, SQL cells)."""
    for task, attack in _cases(world):
        oracle = make_redaction_filter(
            lambda texts, m=attack.marker: [1.0 if m in t else 0.0 for t in texts], threshold=0.5)
        for name, rec, texts in _gold_reads(world, task, attack, output_filter=oracle):
            if not rec.exposed:
                continue
            where = f"{task.template} / {attack.family} / {name}"
            assert rec.filtered and not rec.exposed_after_filter, where
            assert not any(attack.marker in t for t in texts), where
            assert any(REDACTION_NOTICE in t for t in texts), where


def test_detector_dataset_extracts_the_payload_for_every_template(world):
    """Every attacked replay yields a labeled injected segment. (Rows can still be dropped later
    by design: each ticket or article belongs to one split, claimed by test, then val.)"""
    res = build_dataset(DatasetConfig(world_seeds=(0,), instances_per_template=1, probes=False,
                                      augment=False), worlds={0: world})
    assert res.manifest["attacks_without_exposed_segment"] == []
