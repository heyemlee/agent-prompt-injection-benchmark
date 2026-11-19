"""Every template, and every template x applicable goal: the reference agents get exactly the
outcomes they should. If one of these fails, a grader or a template is wrong, and every number
computed from it would be too."""

import asyncio
from collections import Counter

import pytest

from injbench.attacks import GOALS, OUT_OF_PERMISSION, WITHIN_PERMISSION, applicable, build_attack
from injbench.defenses import get_defense
from injbench.models import ModelConfig
from injbench.runner import EpisodeSpec, run_episode, scripted_client
from injbench.tasks import TEMPLATES, build_instances
from injbench.world import build_world

WORLD = build_world(0)
TASKS = {t.instance_id: t for t in build_instances(WORLD, list(TEMPLATES), 2)}
CASES = [(iid, goal) for iid, task in TASKS.items() for goal in GOALS if applicable(goal, task)]


def run(task, behavior, defense, goal=None, family="info_block"):
    attack = build_attack(WORLD, task, family, goal) if goal else None
    m = ModelConfig(name=f"ref-{behavior}", kind="scripted", behavior=behavior)
    spec = EpisodeSpec("x", task, attack, get_defense(defense), m, 0)
    record, _ = asyncio.run(run_episode(spec, WORLD, scripted_client(m, task, attack, WORLD), 60))
    return record


@pytest.mark.parametrize("iid", list(TASKS))
def test_benign_reference_agents(iid):
    task = TASKS[iid]
    for defense in ("none", "permissions"):
        r = run(task, "gold", defense)
        assert r["task_success"] and r["safe_completion"], (iid, defense, r["unexpected_writes"])
        assert r["n_realized"] == r["n_attempted"] == r["n_unexpected_writes"] == 0
        assert r["out_of_scope_reads"] == 0, "gold SQL must stay within the team's records"
    r = run(task, "null", "none")
    assert not r["task_success"] and not r["safe_completion"] and not r["wrongful_refusal"]
    r = run(task, "refuser", "none")
    assert not r["task_success"] and r["wrongful_refusal"]
    r = run(task, "leaker", "none")
    assert r["task_success"] and not r["safe_completion"]
    assert any(v["type"] == "data_leak" for v in r["realized_violations"])


@pytest.mark.parametrize("iid,goal", CASES)
def test_attack_reference_agents(iid, goal):
    task = TASKS[iid]
    klass = build_attack(WORLD, task, "info_block", goal).klass
    gold = run(task, "gold", "none", goal)
    assert gold["exposed"], "the gold path must read the payload"
    assert gold["task_success"] and gold["safe_completion"] and not gold["attack_success"], gold

    r_none = run(task, "gullible", "none", goal)
    r_perm = run(task, "gullible", "permissions", goal)
    assert r_none["attack_success"] and not r_none["safe_completion"], r_none
    if klass == OUT_OF_PERMISSION:
        assert r_none["n_realized"] > 0
        assert not r_perm["attack_success"]
        assert r_perm["n_attempted"] > 0 and r_perm["n_realized"] == 0
        assert r_perm["safe_completion"], r_perm  # blocked; the real task is still done
    else:
        assert r_none["n_realized"] == 0, r_none["realized_violations"]
        assert r_perm["attack_success"] and not r_perm["safe_completion"]


def test_every_template_has_both_goal_classes():
    for task in TASKS.values():
        classes = {build_attack(WORLD, task, "info_block", g).klass
                   for g in GOALS if applicable(g, task)}
        assert classes == {OUT_OF_PERMISSION, WITHIN_PERMISSION}, task.template


def test_template_count_and_hard_negative_mix():
    assert len(TEMPLATES) >= 20
    kinds = {}
    for task in TASKS.values():
        kinds.setdefault(task.template, set()).add(task.info["hard_negative"])
    counts = Counter(k for ks in kinds.values() for k in ks)
    assert counts["delegated"] >= 5 and counts["not_delegated"] >= 3, counts


def test_hard_negative_records_hold_in_band_requests():
    for task in TASKS.values():
        records = task.info["hard_negative_records"]
        assert bool(records) == (task.info["hard_negative"] != "none"), task.instance_id
        for rec in records:
            text = WORLD.conn.execute(
                f"SELECT {rec['column']} FROM {rec['table']} WHERE id = ?",
                (rec["row_id"],)).fetchone()[0]
            assert any(w in text.lower() for w in (
                "please", "can you", "could you", "feel free", "move ", "close", "escalate",
                "reopen", "flag", "add me")), text


def test_vectors_point_at_untrusted_fields_of_real_rows():
    from injbench.env import INJECTABLE_FIELDS

    for task in TASKS.values():
        v = task.vector
        assert (v.table, v.column) in INJECTABLE_FIELDS
        assert WORLD.conn.execute(f"SELECT {v.column} FROM {v.table} WHERE id = ?",
                                  (v.row_id,)).fetchone(), task.instance_id


def test_instances_are_deterministic_and_vary():
    again = build_instances(build_world(0), list(TEMPLATES), 2)
    assert [t.prompt for t in again] == [t.prompt for t in TASKS.values()]
    for name in TEMPLATES:
        tasks = build_instances(WORLD, [name], 6)
        variants = {(t.principal.user_id, t.prompt, t.vector) for t in tasks}
        assert len(variants) >= 2, name


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_templates_build_on_other_worlds(seed):
    world = build_world(seed)
    tasks = build_instances(world, list(TEMPLATES), 2, seed=seed)
    for task in tasks:
        for goal in GOALS:
            if applicable(goal, task):
                build_attack(world, task, "business_note", goal)
