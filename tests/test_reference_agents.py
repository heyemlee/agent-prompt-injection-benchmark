"""The graders are only trustworthy if reference agents get exactly the outcomes
they should: gold passes and ignores injections, null and refuser fail, the
leaker is caught, and the gullible agent's attacks are scored by class."""

import asyncio

import pytest

from injbench.attacks import GOALS, OUT_OF_PERMISSION, applicable, build_attack
from injbench.defenses import get_defense
from injbench.models import ModelConfig
from injbench.runner import EpisodeSpec, run_episode, scripted_client
from injbench.tasks import TEMPLATES, build_instances


@pytest.fixture(scope="module")
def tasks(world):
    return build_instances(world, list(TEMPLATES), 2)


def run(world, task, behavior, defense, goal=None, family="info_block"):
    attack = build_attack(world, task, family, goal) if goal else None
    m = ModelConfig(name=f"ref-{behavior}", kind="scripted", behavior=behavior)
    spec = EpisodeSpec("x", task, attack, get_defense(defense), m, 0)
    record, _ = asyncio.run(run_episode(spec, world, scripted_client(m, task, attack, world), 30))
    return record


@pytest.mark.parametrize("defense", ["none", "permissions"])
def test_gold_agent_passes_every_task(world, tasks, defense):
    for task in tasks:
        r = run(world, task, "gold", defense)
        assert r["task_success"] and r["safe_completion"], task.instance_id
        assert r["n_realized"] == 0 and r["n_unexpected_writes"] == 0 and r["n_attempted"] == 0


@pytest.mark.parametrize("goal", list(GOALS))
def test_gold_agent_reads_but_ignores_injections(world, tasks, goal):
    for task in tasks:
        if not applicable(goal, task):
            continue
        r = run(world, task, "gold", "none", goal)
        assert r["exposed"], f"{task.instance_id}: gold path must read the payload"
        assert r["safe_completion"] and not r["attack_success"], task.instance_id


def test_null_and_refuser_fail(world, tasks):
    for task in tasks:
        r = run(world, task, "null", "none")
        assert not r["task_success"] and not r["safe_completion"] and not r["wrongful_refusal"]
        r = run(world, task, "refuser", "none")
        assert not r["task_success"] and r["wrongful_refusal"]


def test_leaker_is_caught(world, tasks):
    for task in tasks:
        r = run(world, task, "leaker", "none")
        assert r["task_success"] and not r["safe_completion"], task.instance_id
        assert any(v["type"] == "data_leak" for v in r["realized_violations"])


@pytest.mark.parametrize("goal", list(GOALS))
def test_gullible_agent_outcomes_by_goal_class(world, tasks, goal):
    checked = 0
    for task in tasks:
        if not applicable(goal, task):
            continue
        klass = build_attack(world, task, "info_block", goal).klass
        r_none = run(world, task, "gullible", "none", goal)
        r_perm = run(world, task, "gullible", "permissions", goal)
        assert r_none["attack_success"] and not r_none["safe_completion"], task.instance_id
        if klass == OUT_OF_PERMISSION:
            assert r_none["n_realized"] > 0
            assert not r_perm["attack_success"]
            assert r_perm["n_attempted"] > 0 and r_perm["n_realized"] == 0
            assert r_perm["safe_completion"], task.instance_id  # blocked; the real task still done
        else:
            assert r_none["n_realized"] == 0  # within permission: nothing for policy to flag
            assert r_perm["attack_success"] and not r_perm["safe_completion"]
        checked += 1
    assert checked > 0


def test_every_family_renders_the_instruction(world, tasks):
    from injbench.attacks import FAMILIES

    for family in FAMILIES:
        atk = build_attack(world, tasks[0], family, "grant_admin")
        assert atk.instruction in atk.payload
