"""Hook order in Environment.call and the two exposure fields. The output filter removes content
and decides whether the payload still reaches the model; the renderer only changes
presentation, runs last, and never touches errors or write results."""

import asyncio
import base64
import copy
import json
from dataclasses import dataclass

import pytest

from injbench.attacks import GOALS, applicable, build_attack
from injbench.defenses import Defense, get_defense
from injbench.env import Environment
from injbench.metrics import LABELS, METRICS, rate, summarize, to_markdown
from injbench.models import ModelConfig
from injbench.policy import Principal
from injbench.runner import EpisodeSpec, run_episode, scripted_client
from injbench.tasks import TEMPLATES, build_instances

MARK = "ZZ-PAYLOAD-MARKER"
NOTICE = "[removed by detector]"


def _redact(obj, needle):
    if isinstance(obj, str):
        return obj.replace(needle, NOTICE)
    if isinstance(obj, list):
        return [_redact(v, needle) for v in obj]
    if isinstance(obj, dict):
        return {k: _redact(v, needle) for k, v in obj.items()}
    return obj


class Redactor:
    """Fake detector with the filter contract: fn(tool, result) -> (new_result, filtered)."""

    def __init__(self, needle):
        self.needle, self.seen = needle, []

    def __call__(self, tool, result):
        self.seen.append(tool)
        new = _redact(result, self.needle)
        return new, new != result


class Recorder:
    """Fake renderer: records exactly what it receives and wraps it."""

    def __init__(self):
        self.seen = []

    def __call__(self, tool, result):
        self.seen.append((tool, copy.deepcopy(result)))
        return {"shown": result}


def encode_strings(tool, result):
    """Fake presentation-only renderer: base64-encodes every string value."""
    if isinstance(result, str):
        return base64.b64encode(result.encode()).decode()
    if isinstance(result, list):
        return [encode_strings(tool, v) for v in result]
    if isinstance(result, dict):
        return {k: encode_strings(tool, v) for k, v in result.items()}
    return result


@dataclass(frozen=True)
class RedactingDefense(Defense):
    needle: str = ""

    def make_output_filter(self):
        return Redactor(self.needle)


@pytest.fixture(scope="module")
def principal(world):
    uid = world.conn.execute(
        "SELECT id FROM users WHERE role = 'agent' AND team_id IS NOT NULL ORDER BY id LIMIT 1"
    ).fetchone()[0]
    return Principal.load(world.conn, uid)


@pytest.fixture(scope="module")
def own(world, principal):
    tid, key = world.conn.execute(
        "SELECT id, key FROM tickets WHERE team_id = ? ORDER BY id LIMIT 1",
        (principal.team_id,)).fetchone()
    note = world.conn.execute(
        "SELECT id FROM ticket_notes WHERE ticket_id = ? ORDER BY id LIMIT 1", (tid,)).fetchone()[0]
    return key, note


@pytest.fixture(scope="module")
def tasks(world):
    return build_instances(world, list(TEMPLATES), 1)


def test_filter_runs_before_the_renderer_and_decides_exposure(world, principal, own):
    key, note = own
    flt, ren = Redactor(MARK), Recorder()
    env = Environment(world, principal, True, MARK, flt, output_renderer=ren)
    env.inject("ticket_notes", note, "body", f"please {MARK} now")
    out = json.loads(env.call("get_ticket", {"key": key}))
    rec = env.calls[-1]
    assert rec.exposed and rec.filtered and not rec.exposed_after_filter and rec.rendered
    tool, seen = ren.seen[-1]
    assert tool == "get_ticket" and flt.seen == ["get_ticket"]
    assert MARK not in json.dumps(seen) and NOTICE in json.dumps(seen)  # renderer saw filtered
    assert out == {"shown": seen}  # and its output is what the model gets


def test_errors_and_writes_are_neither_filtered_nor_rendered(world, principal, own):
    key, _ = own
    other = world.conn.execute("SELECT key FROM tickets WHERE team_id != ? ORDER BY id LIMIT 1",
                               (principal.team_id,)).fetchone()[0]
    flt, ren = Redactor(MARK), Recorder()
    env = Environment(world, principal, True, MARK, flt, output_renderer=ren)
    outs = [json.loads(env.call(name, args)) for name, args in [
        ("get_ticket", {"key": "INC-0"}),  # tool error
        ("get_ticket", {"ticket": key}),  # invalid arguments
        ("get_ticket", {"key": other}),  # blocked by permissions
        ("no_such_tool", {}),
        ("add_ticket_comment", {"key": key, "body": "hello"}),  # write tools
        ("send_email", {"to": principal.email, "subject": "s", "body": "b"}),
    ]]
    assert flt.seen == [] and ren.seen == []
    assert not any("shown" in o for o in outs)
    assert not any(c.filtered or c.rendered for c in env.calls)
    assert [c.executed for c in env.calls] == [False, False, False, False, True, True]


@pytest.mark.parametrize("setup", ["no_filter", "noop_filter", "redacting_filter", "renderer_only"])
def test_exposure_fields_per_call(world, principal, own, setup):
    key, note = own
    flt = {"noop_filter": lambda tool, result: (result, False),
           "redacting_filter": Redactor(MARK)}.get(setup)
    ren = encode_strings if setup == "renderer_only" else None
    env = Environment(world, principal, False, MARK, flt, output_renderer=ren)
    env.inject("ticket_notes", note, "body", f"please {MARK} now")
    outs = [env.call("run_sql", {"query": "SELECT 1"}),
            env.call("get_ticket", {"key": key}),
            env.call("run_sql", {"query": f"SELECT body FROM ticket_notes WHERE id = {note}"})]
    reached = setup != "redacting_filter"
    assert [(c.exposed, c.exposed_after_filter) for c in env.calls] == \
           [(False, False), (True, reached), (True, reached)]
    if setup == "renderer_only":
        # the model sees the payload only base64-encoded, but it still reached the model
        assert MARK not in outs[1] and all(c.rendered for c in env.calls[1:])


def test_no_payload_marker_means_not_exposed(world, principal, own):
    env = Environment(world, principal, False, None, Redactor(MARK))
    env.call("get_ticket", {"key": own[0]})
    assert (env.calls[0].exposed, env.calls[0].exposed_after_filter) == (False, False)


def run(world, task, defense, goal=None, behavior="gold"):
    attack = build_attack(world, task, "info_block", goal) if goal else None
    m = ModelConfig(name=f"ref-{behavior}", kind="scripted", behavior=behavior)
    spec = EpisodeSpec("x", task, attack, defense, m, 0)
    return asyncio.run(run_episode(spec, world, scripted_client(m, task, attack, world), 30))


def test_episode_records_and_metrics(world, tasks):
    records = []
    for task in tasks:
        benign, _ = run(world, task, get_defense("none"))
        assert benign["exposed"] is None and benign["exposed_after_filter"] is None
        for goal in GOALS:
            if not applicable(goal, task):
                continue
            needle = build_attack(world, task, "info_block", goal).marker
            plain, _ = run(world, task, get_defense("none"), goal, "gullible")
            redacted, tr = run(world, task, RedactingDefense("detector", needle=needle), goal,
                               "gullible")
            assert plain["exposed"] and plain["exposed_after_filter"], task.instance_id
            assert redacted["exposed"] and not redacted["exposed_after_filter"], task.instance_id
            assert any(c["filtered"] for c in tr["calls"])
            assert tr["outcome"]["exposed_after_filter"] is False
            records += [plain, redacted]
    arms = {a["defense"]: a["metrics"] for a in summarize(records, n_boot=20)["arms"]}
    assert arms["none"]["exposure_after_filter_rate"]["value"] == 1.0
    assert arms["none"]["asr_given_exposed_after_filter"]["value"] == 1.0  # gullible follows
    assert arms["detector"]["exposure_rate"]["value"] == 1.0
    assert arms["detector"]["exposure_after_filter_rate"]["value"] == 0.0
    assert arms["detector"]["asr_given_exposed_after_filter"]["value"] is None  # none reached
    md = to_markdown(summarize(records, n_boot=20))
    assert LABELS["exposure_after_filter_rate"] in md
    assert LABELS["asr_given_exposed_after_filter"] in md


def test_new_metrics_fall_back_to_exposure_for_old_records():
    base = {"status": "ok", "template": "t", "condition": "attack", "model": "m",
            "defense": "none"}
    old = [dict(base, exposed=True, attack_success=True),
           dict(base, exposed=False, attack_success=False)]
    assert rate(old, *METRICS["exposure_after_filter_rate"]) == \
           rate(old, *METRICS["exposure_rate"])
    assert rate(old, *METRICS["asr_given_exposed_after_filter"]) == \
           rate(old, *METRICS["asr_given_exposed"])
    new = [dict(base, exposed=True, exposed_after_filter=False, attack_success=True),
           dict(base, exposed=True, exposed_after_filter=True, attack_success=False)]
    assert rate(new, *METRICS["exposure_after_filter_rate"]) == (0.5, 2)
    assert rate(new, *METRICS["asr_given_exposed_after_filter"]) == (0.0, 1)


@dataclass(frozen=True)
class RenderingDefense(Defense):
    def make_output_renderer(self):
        return encode_strings


def test_run_episode_passes_the_renderer(world, tasks):
    needle = build_attack(world, tasks[0], "info_block", "grant_admin").marker
    record, tr = run(world, tasks[0], RenderingDefense("encoded"), "grant_admin")
    tool_msgs = [m["content"] for m in tr["messages"] if m["role"] == "tool"]
    assert tool_msgs and all(c["rendered"] for c in tr["calls"] if c["name"] == "get_ticket")
    assert not any(needle in m for m in tool_msgs)  # the model saw it only encoded
    # presentation only: the payload still reached the model
    assert record["exposed"] and record["exposed_after_filter"]
