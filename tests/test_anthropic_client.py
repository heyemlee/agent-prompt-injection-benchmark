"""Offline tests for the Anthropic backend: a fake SDK object stands in for the API."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from injbench.anthropic_client import AnthropicClient, to_anthropic_messages, to_anthropic_tools
from injbench.attacks import build_attack
from injbench.defenses import get_defense
from injbench.env import TOOL_SPECS
from injbench.models import Budget, BudgetExceeded, ModelConfig, ResponseCache
from injbench.runner import EpisodeSpec, run_episode
from injbench.tasks import build_instances


class FakeMessage:
    def __init__(self, data):
        self.data = data

    def to_dict(self):
        return json.loads(json.dumps(self.data))


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def create(self, **params):
        self.requests.append(json.loads(json.dumps(params)))
        return FakeMessage(self.responses.pop(0))


def usage(inp=100, out=50, write=1000, read=2000):
    return {"input_tokens": inp, "output_tokens": out, "cache_creation_input_tokens": write,
            "cache_read_input_tokens": read}


def tool_use(name, args, tid="toolu_1"):
    return {"content": [{"type": "thinking", "thinking": "", "signature": "sig-1"},
                        {"type": "tool_use", "id": tid, "name": name, "input": args}],
            "stop_reason": "tool_use", "usage": usage()}


def final(text):
    return {"content": [{"type": "text", "text": text, "citations": None}],
            "stop_reason": "end_turn", "usage": usage(write=0)}


@pytest.fixture()
def make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_ANTHROPIC_KEY", "x")

    def make(responses, budget=10.0):
        cfg = ModelConfig(name="claude-test", kind="anthropic", model="claude-haiku-5-5",
                          api_key_env="FAKE_ANTHROPIC_KEY", price_in_per_m=1.0,
                          price_out_per_m=10.0, price_cache_write_per_m=1.25,
                          price_cache_read_per_m=0.1)
        client = AnthropicClient(cfg, ResponseCache(tmp_path / "cache.sqlite"), Budget(budget))
        fake = FakeMessages(responses)
        client.client = SimpleNamespace(messages=fake)
        return client, fake

    return make


def test_custom_ca_bundle_is_opt_in(monkeypatch):
    import os

    from injbench.anthropic_client import custom_ca_http_client

    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    assert custom_ca_http_client() is None
    if os.path.exists("/etc/ssl/cert.pem"):
        monkeypatch.setenv("SSL_CERT_FILE", "/etc/ssl/cert.pem")
        assert custom_ca_http_client() is not None


def test_tool_specs_convert():
    tools = to_anthropic_tools(TOOL_SPECS)
    assert len(tools) == len(TOOL_SPECS)
    assert set(tools[0]) == {"name", "description", "input_schema"}


def test_messages_convert_and_group_tool_results():
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "t1", "function": {"name": "run_sql", "arguments": '{"query": "SELECT 1"}'}},
            {"id": "t2", "function": {"name": "get_ticket", "arguments": '{"key": "INC-1"}'}}]},
        {"role": "tool", "tool_call_id": "t1", "content": '{"rows": []}'},
        {"role": "tool", "tool_call_id": "t2", "content": '{"error": "ticket not found"}'},
        {"role": "assistant", "content": "done", "anthropic_content": [
            {"type": "thinking", "thinking": "", "signature": "s"}, {"type": "text", "text": "done"}]},
    ]
    system, out = to_anthropic_messages(msgs)
    assert system == "SYS"
    assert [m["role"] for m in out] == ["user", "assistant", "user", "assistant"]
    assert out[1]["content"][0] == {"type": "tool_use", "id": "t1", "name": "run_sql",
                                    "input": {"query": "SELECT 1"}}
    results = out[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]  # one user message for both
    assert results[1]["is_error"] is True and "is_error" not in results[0]
    assert out[3]["content"][0] == {"type": "thinking", "thinking": "", "signature": "s"}


def test_request_shape_cost_and_local_cache(make_client):
    client, fake = make_client([tool_use("run_sql", {"query": "SELECT 1"})])
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "go"}]
    r = asyncio.run(client.complete(msgs, TOOL_SPECS))
    req = fake.requests[0]
    assert req["system"] == [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}]
    assert req["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in req and req["max_tokens"] == 16000
    assert r.message["tool_calls"][0]["function"]["name"] == "run_sql"
    assert json.loads(r.message["tool_calls"][0]["function"]["arguments"]) == {"query": "SELECT 1"}
    assert r.message["anthropic_content"][0] == {"type": "thinking", "thinking": "",
                                                 "signature": "sig-1"}
    expected = (100 * 1.0 + 1000 * 1.25 + 2000 * 0.1 + 50 * 10.0) / 1e6
    assert r.cost_usd == pytest.approx(expected)
    assert client.budget.spent == pytest.approx(expected)
    assert (r.cache_write_tokens, r.cache_read_tokens) == (1000, 2000)
    again = asyncio.run(client.complete(msgs, TOOL_SPECS))
    assert again.cached and len(fake.requests) == 1  # replayed from the local cache


def test_refusal_max_tokens_and_budget(make_client):
    refusal = {"content": [], "stop_reason": "refusal",
               "stop_details": {"type": "refusal", "category": "cyber", "explanation": None},
               "usage": usage()}
    cut = {"content": [{"type": "tool_use", "id": "t", "name": "run_sql", "input": {}}],
           "stop_reason": "max_tokens", "usage": usage()}
    client, _ = make_client([refusal, cut], budget=10.0)
    r = asyncio.run(client.complete([{"role": "user", "content": "a"}], TOOL_SPECS))
    assert r.message["refusal"]["category"] == "cyber" and "tool_calls" not in r.message
    r = asyncio.run(client.complete([{"role": "user", "content": "b"}], TOOL_SPECS))
    assert "tool_calls" not in r.message  # an incomplete call is never executed
    client.budget.spent = client.budget.limit
    with pytest.raises(BudgetExceeded):
        asyncio.run(client.complete([{"role": "user", "content": "c"}], TOOL_SPECS))


def test_full_episode_through_the_agent_loop(world, make_client):
    task = build_instances(world, ["kb_answer_comment"], 1)[0]
    code = task.info["kb_code"]
    key = task.gold_actions[-1][1]["key"]
    client, fake = make_client([
        tool_use("get_kb_article", {"code": code}, "toolu_a"),
        tool_use("add_ticket_comment", {"key": key, "body": f"See {code}."}, "toolu_b"),
        final("Done."),
    ])
    attack = build_attack(world, task, "info_block", "grant_admin")
    spec = EpisodeSpec("e", task, attack, get_defense("permissions"), client.cfg, 0)
    record, transcript = asyncio.run(run_episode(spec, world, client, 15))
    assert record["task_success"] and record["safe_completion"] and record["exposed"]
    assert record["cost_usd"] > 0 and record["cache_read_tokens"] == 6000
    second = fake.requests[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[1]["content"][0]["signature"] == "sig-1"  # thinking block round-trips
    assert second[2]["content"][0]["type"] == "tool_result"
    assert attack.instruction in second[2]["content"][0]["content"]  # the payload reached the model


def test_model_refusal_counts_as_wrongful_refusal_on_benign_tasks(world, make_client):
    task = build_instances(world, ["sla_summary_report"], 1)[0]
    refusal = {"content": [{"type": "text", "text": ""}], "stop_reason": "refusal",
               "stop_details": {"type": "refusal", "category": None}, "usage": usage()}
    client, _ = make_client([refusal])
    spec = EpisodeSpec("e", task, None, get_defense("none"), client.cfg, 0)
    record, _ = asyncio.run(run_episode(spec, world, client, 15))
    assert record["model_refused"] and record["wrongful_refusal"] and not record["task_success"]
