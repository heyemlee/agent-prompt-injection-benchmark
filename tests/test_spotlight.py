"""Spotlighting codec and defense arms: every mode round-trips exactly, only untrusted values
are marked, and the per-episode marker is deterministic and consistent with the prompt."""

import asyncio
import base64
import json
import re

import pytest

from injbench.agent import system_prompt
from injbench.attacks import build_attack
from injbench.cli import main
from injbench.defenses import DEFENSES, Defense, SpotlightDefense, get_defense
from injbench.env import Environment
from injbench.models import ModelConfig
from injbench.policy import Principal
from injbench.runner import EpisodeSpec, episode_seed, run_episode, scripted_client
from injbench.spotlight import (B64_PREFIX, DATAMARK_CHARS, MODES, Spotlighter, choose_marker,
                                is_free_text, unspotlight)
from injbench.tasks import TEMPLATES, build_instances

SPOTLIGHT_ARMS = [n for n, d in DEFENSES.items() if isinstance(d, SpotlightDefense)]

SAMPLES = [
    "Confirmed fixed on my side, you can close this ticket.",
    "Still happening, any update?\n\nSecond paragraph\twith a tab.",
    "  leading and trailing  ",
    "line one\r\nline two \n indented",
    "non-breaking space and CJK: 本周共有5个SLA超时工单 - done",
    "INC-1047 and KB-2345 and a.b@acme.example on 2026-11-09",
    "single",
    "",
]


def all_spotlighters():
    for mode in MODES:
        if mode == "datamark":
            yield from (Spotlighter(mode, ch) for ch in DATAMARK_CHARS)
        else:
            yield Spotlighter(mode, choose_marker(mode, "seed"))


@pytest.mark.parametrize("sp", list(all_spotlighters()), ids=lambda sp: f"{sp.mode}-{sp.marker}")
def test_every_mode_round_trips(sp):
    for text in SAMPLES:
        marked = sp.mark(text)
        assert sp.unmark(marked) == text
        assert unspotlight(marked) == text  # the graders' marker-agnostic inverse
        # datamarking has nothing to mark in a single word; empty values stay empty in every mode
        expect_change = bool(text) and (sp.mode != "datamark" or any(c.isspace() for c in text))
        assert (marked != text) == expect_change


def test_base64_values_decode():
    sp = Spotlighter("base64")
    for text in SAMPLES[:-1]:
        marked = sp.mark(text)
        assert marked.startswith(B64_PREFIX)
        assert base64.b64decode(marked[len(B64_PREFIX):], validate=True).decode() == text


def test_datamark_marks_every_whitespace():
    for ch in DATAMARK_CHARS:
        sp = Spotlighter("datamark", ch)
        for text in SAMPLES:
            marked = sp.mark(text)
            assert " " not in marked
            assert marked.count(ch) == sum(c.isspace() for c in text)
            # other whitespace (line breaks, tabs) stays, with the marker in front of it
            assert all(marked[i - 1] == ch for i, c in enumerate(marked) if c.isspace())


def test_delimit_wraps_with_the_episode_id():
    sp = Spotlighter("delimit", choose_marker("delimit", "x"))
    marked = sp.mark("hello world")
    assert marked == f"<<data:{sp.marker}>>hello world<</data:{sp.marker}>>"
    assert re.fullmatch(r"[0-9a-f]{8}", sp.marker)


def test_markers_are_deterministic_per_seed():
    seeds = [f"sla_summary_report-{i}|a{a}" for i in range(20) for a in range(2)]
    for mode in MODES:
        assert [choose_marker(mode, s) for s in seeds] == [choose_marker(mode, s) for s in seeds]
    assert len({choose_marker("datamark", s) for s in seeds}) > 1
    assert len({choose_marker("delimit", s) for s in seeds}) == len(seeds)
    assert {choose_marker("datamark", s) for s in seeds} <= set(DATAMARK_CHARS)
    assert choose_marker("base64", "anything") == ""
    with pytest.raises(ValueError):
        choose_marker("rot13", "x")


def _world_texts(world):
    queries = ("SELECT title FROM tickets", "SELECT body FROM ticket_notes",
               "SELECT title FROM kb_articles", "SELECT body FROM kb_articles",
               "SELECT body FROM reports", "SELECT name FROM users", "SELECT name FROM teams")
    return [t for q in queries for (t,) in world.conn.execute(q)]


def test_world_text_round_trips_and_has_no_marks(world):
    texts = _world_texts(world)
    assert len(texts) > 1000
    for text in texts:
        assert unspotlight(text) == text  # no spotlighting artifacts in the world itself
        assert not any(ch in text for ch in DATAMARK_CHARS)
        for sp in all_spotlighters():
            assert unspotlight(sp.mark(text)) == text


def test_free_text_classification():
    free = ["Confirmed fixed, you can close this ticket.", "Facilities IT", "Avery Chen",
            "Wi-Fi slow on 3rd floor", "x" * 41]
    structured = ["INC-1047", "KB-2345", "avery.chen@acme.example", "2026-11-09",
                  "2026-11-09T12:00", "2026-11-09 12:00:30", "+1 415-555-0100", "(415) 555-0100",
                  "42", "3.5", "open", "P1", "network-l1", "Network", "", "   "]
    assert all(is_free_text(s) for s in free)
    assert not any(is_free_text(s) for s in structured)


def test_unspotlight_leaves_plain_output_alone():
    plain = ["SLA summary, week of 2026-11-09: 3 tickets breached SLA.\n- INC-1047: noted",
             "Triaged: INC-1001 -> network-l2, INC-1002 → network-oncall",
             "Commit aee8fd8c0ffee1234567890abcdef1234 and Internationalization done",
             "本周共有5个SLA超时工单因故延误", "a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p", "<<not a tag>>",
             "b64:not-base64!"]
    for text in plain:
        assert unspotlight(text) == text


@pytest.fixture(scope="module")
def principal(world):
    uid = world.conn.execute(
        "SELECT id FROM users WHERE role = 'agent' AND team_id IS NOT NULL ORDER BY id LIMIT 1"
    ).fetchone()[0]
    return Principal.load(world.conn, uid)


def _calls(world, principal):
    key = world.conn.execute("SELECT key FROM tickets WHERE team_id = ? ORDER BY id LIMIT 1",
                             (principal.team_id,)).fetchone()[0]
    code = world.conn.execute("SELECT code FROM kb_articles ORDER BY id LIMIT 1").fetchone()[0]
    return [
        ("get_ticket", {"key": key}),
        ("get_kb_article", {"code": code}),
        ("search_kb", {"query": "reset VPN profile"}),
        ("run_sql", {"query": "SELECT id, key, title, priority, status, queue, created_date, "
                              "due_date, sla_breached FROM tickets ORDER BY id LIMIT 20"}),
        ("run_sql", {"query": "SELECT id, name, email, phone, team_id FROM users ORDER BY id"}),
        ("run_sql", {"query": "SELECT n.id, n.ticket_id, n.author_kind, n.created_at, n.body "
                              "FROM ticket_notes n ORDER BY n.id LIMIT 30"}),
    ]


def _unmark_all(obj, sp):
    if isinstance(obj, str):
        return sp.unmark(obj)
    if isinstance(obj, list):
        return [_unmark_all(v, sp) for v in obj]
    if isinstance(obj, dict):
        return {k: _unmark_all(v, sp) for k, v in obj.items()}
    return obj


UNTRUSTED = {"get_ticket": {"title"}, "get_kb_article": {"title", "body"}}


@pytest.mark.parametrize("arm", ["spotlight_delimit", "spotlight_datamark", "spotlight_base64"])
def test_rendering_marks_untrusted_values_only(world, principal, arm):
    d = get_defense(arm).for_episode("seed")
    sp = d.make_output_renderer()
    raw_env = Environment(world, principal, enforce_permissions=False)
    env = Environment(world, principal, enforce_permissions=False, output_renderer=sp)
    for tool, args in _calls(world, principal):
        raw = json.loads(raw_env.call(tool, args))
        shown = json.loads(env.call(tool, args))
        assert env.calls[-1].rendered
        assert _unmark_all(shown, sp) == raw  # nothing lost, exact round trip through JSON
        if tool == "run_sql":
            cols = raw["columns"]
            for raw_row, row in zip(raw["rows"], shown["rows"]):
                for col, a, b in zip(cols, raw_row, row):
                    if col in ("body", "title", "name"):
                        assert a != b, (col, a)  # free text (multi-word names too) is marked
                    else:  # ids, keys, emails, phones, dates, statuses: untouched
                        assert a == b, (col, a)
        elif tool == "search_kb":
            assert [r["code"] for r in shown["results"]] == [r["code"] for r in raw["results"]]
            assert all(a["title"] != b["title"] for a, b in zip(raw["results"], shown["results"]))
        else:
            for field, value in raw.items():
                if field in UNTRUSTED[tool]:
                    assert shown[field] != value
                elif field == "notes":
                    for a, b in zip(value, shown["notes"]):
                        assert a["body"] != b["body"]
                        assert {k: v for k, v in a.items() if k != "body"} == \
                               {k: v for k, v in b.items() if k != "body"}
                else:
                    assert shown[field] == value, field


def test_registry_and_cli_list(capsys):
    expected = {"spotlight_delimit", "spotlight_datamark", "spotlight_base64",
                "permissions+spotlight_delimit", "permissions+spotlight_datamark",
                "permissions+spotlight_base64"}
    assert expected == set(SPOTLIGHT_ARMS)
    for name in expected:
        d = get_defense(name)
        assert d.enforce_permissions == name.startswith("permissions+")
        assert d.make_output_filter() is None
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in expected)


def test_base_defense_hooks_default_to_nothing():
    d = Defense("x")
    assert d.make_output_renderer() is None and d.for_episode("s") is d
    assert get_defense("permissions").for_episode("s") is get_defense("permissions")
    with pytest.raises(ValueError):
        SpotlightDefense("bad", mode="rot13")


@pytest.mark.parametrize("arm", SPOTLIGHT_ARMS)
def test_addendum_matches_the_rendered_marker(arm):
    for d in (get_defense(arm), get_defense(arm).for_episode("sla_summary_report-0|a0")):
        sp = d.make_output_renderer()
        assert sp.mode == d.mode and sp.marker == d.marker
        shown = sp.mark("hello world")
        if d.mode == "delimit":
            assert sp.open_tag in d.system_addendum and sp.close_tag in d.system_addendum
            assert shown.startswith(sp.open_tag)
        elif d.mode == "datamark":
            assert f"Hello{d.marker}world" in d.system_addendum and d.marker in shown
        else:
            assert B64_PREFIX in d.system_addendum and shown.startswith(B64_PREFIX)
        assert "never as instructions" in d.system_addendum


def test_for_episode_is_deterministic_and_shared_across_permission_variants(world):
    task = build_instances(world, list(TEMPLATES), 1)[0]
    for mode in MODES:
        plain, perm = get_defense(f"spotlight_{mode}"), get_defense(f"permissions+spotlight_{mode}")
        a, b = plain.for_episode("s|a0"), perm.for_episode("s|a0")
        assert a == plain.for_episode("s|a0") and a.marker == choose_marker(mode, "s|a0")
        assert a.name == plain.name and b.enforce_permissions
        # same prompt, so the two arms share model samples until a call is blocked
        assert system_prompt(world, task, a) == system_prompt(world, task, b)
        assert system_prompt(world, task, a).endswith(a.system_addendum)


@pytest.mark.parametrize("arm", ["spotlight_delimit", "spotlight_datamark", "spotlight_base64"])
def test_run_episode_spotlights_with_the_episode_marker(world, arm):
    task = build_instances(world, list(TEMPLATES), 1)[0]
    attack = build_attack(world, task, "info_block", "grant_admin")
    m = ModelConfig(name="ref-gold", kind="scripted", behavior="gold")
    spec = EpisodeSpec("x", task, attack, get_defense(arm), m, 0)
    record, tr = asyncio.run(run_episode(spec, world, scripted_client(m, task, attack, world), 30))
    d = get_defense(arm).for_episode(episode_seed(spec))
    assert tr["messages"][0]["content"].endswith(d.system_addendum)
    tool_msgs = [msg["content"] for msg in tr["messages"] if msg["role"] == "tool"]
    expected = {"delimit": d.make_output_renderer().open_tag, "datamark": d.marker,
                "base64": B64_PREFIX}[d.mode]
    assert any(expected in msg for msg in tool_msgs)
    assert any(c["rendered"] for c in tr["calls"])
    # spotlighting changes presentation only: the payload still reached the model
    assert record["exposed"] and record["exposed_after_filter"]
