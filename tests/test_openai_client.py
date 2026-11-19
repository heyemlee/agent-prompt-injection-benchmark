"""Offline tests for the OpenAI-compatible backend (OpenRouter, Together, local servers...)."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from injbench.models import Budget, ModelConfig, OpenAICompatClient, ResponseCache, sanitize_messages
from injbench.runner import load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


class FakeCompletions:
    def __init__(self, finish="tool_calls", reasoning=None, provider=None):
        self.finish, self.reasoning, self.provider = finish, reasoning, provider
        self.requests = []

    async def create(self, **params):
        self.requests.append(params)
        call = SimpleNamespace(id="c1", function=SimpleNamespace(
            name="run_sql", arguments='{"query": "SELECT 1"}'))
        msg = SimpleNamespace(content=None, tool_calls=[call],
                              model_extra={"reasoning": self.reasoning} if self.reasoning else {})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason=self.finish)],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=100),
            model_extra={"provider": self.provider} if self.provider else {})


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_OR_KEY", "x")

    def make(fake):
        cfg = ModelConfig(name="q", model="qwen/qwen3.5-9b", api_key_env="FAKE_OR_KEY",
                          price_in_per_m=0.17, price_out_per_m=0.25)
        c = OpenAICompatClient(cfg, ResponseCache(tmp_path / "c.sqlite"), Budget(1.0))
        c.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
        return c

    return make


def test_sanitize_drops_bookkeeping_fields():
    msgs = [{"role": "assistant", "content": None, "tool_calls": [], "stop_reason": "x",
             "reasoning": "...", "provider": "p", "anthropic_content": []},
            {"role": "tool", "tool_call_id": "c1", "content": "{}"}]
    assert sanitize_messages(msgs) == [{"role": "assistant", "content": None, "tool_calls": []},
                                       {"role": "tool", "tool_call_id": "c1", "content": "{}"}]


def test_reasoning_and_provider_are_recorded_but_not_sent(client):
    fake = FakeCompletions(reasoning="thinking...", provider="DeepInfra")
    c = client(fake)
    history = [{"role": "user", "content": "go"},
               {"role": "assistant", "content": "x", "reasoning": "old", "stop_reason": "stop"}]
    r = asyncio.run(c.complete(history, []))
    assert r.message["reasoning"] == "thinking..." and r.message["provider"] == "DeepInfra"
    assert r.message["tool_calls"][0]["function"]["name"] == "run_sql"
    sent = fake.requests[0]["messages"]
    assert all(set(m) <= {"role", "content", "tool_calls", "tool_call_id", "name"} for m in sent)
    assert r.cost_usd == pytest.approx((1000 * 0.17 + 100 * 0.25) / 1e6)


def test_identical_concurrent_requests_share_one_call(client):
    class SlowCompletions(FakeCompletions):
        async def create(self, **params):
            await asyncio.sleep(0.05)
            return await super().create(**params)

    fake = SlowCompletions()
    c = client(fake)
    msgs = [{"role": "user", "content": "same prompt"}]

    async def both():
        return await asyncio.gather(c.complete(msgs, []), c.complete(msgs, []))

    a, b = asyncio.run(both())
    assert len(fake.requests) == 1  # one API call, one sample
    assert a.message == b.message and sorted([a.cached, b.cached]) == [False, True]
    assert c.budget.spent == pytest.approx(a.cost_usd)  # charged once


def test_siliconflow_reasoning_content_is_recorded(client):
    class SFCompletions(FakeCompletions):
        async def create(self, **params):
            resp = await super().create(**params)
            resp.choices[0].message.model_extra = {"reasoning_content": "thought"}
            return resp

    r = asyncio.run(client(SFCompletions()).complete([{"role": "user", "content": "go"}], []))
    assert r.message["reasoning"] == "thought"


def test_truncated_and_filtered_turns(client):
    r = asyncio.run(client(FakeCompletions(finish="length")).complete([{"role": "user", "content": "a"}], []))
    assert "tool_calls" not in r.message  # never execute a call that may be cut off
    r = asyncio.run(client(FakeCompletions(finish="content_filter")).complete([{"role": "user", "content": "b"}], []))
    assert r.message["refusal"] == {"type": "content_filter"}


def test_qwen_configs_parse():
    cfg = load_config(CONFIGS / "qwen_openrouter.toml")
    m = cfg.models[0]
    assert m.kind == "openai_compat" and m.base_url.startswith("https://openrouter.ai")
    prov = m.extra_params["extra_body"]["provider"]
    assert prov["quantizations"] == ["bf16"] and prov["require_parameters"] is True
    sf = load_config(CONFIGS / "qwen_siliconflow.toml").models[0]
    assert sf.base_url == "https://api.siliconflow.com/v1" and sf.api_key_env == "SILICONFLOW_API_KEY"
    assert sf.extra_params == {"max_tokens": 8192, "extra_body": {"enable_thinking": False}}
    tinker = load_config(CONFIGS / "qwen_tinker.toml").models[0]
    assert tinker.kind == "anthropic" and tinker.api_key_env == "TINKER_API_KEY"
    assert tinker.temperature is None and not tinker.extra_params  # no effort/temperature sent


def test_max_retries_is_configurable(tmp_path, monkeypatch):
    """Providers with per-minute token quotas need retries that outlast the quota window."""
    monkeypatch.setenv("FAKE_OR_KEY", "x")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)

    def make(**kw):
        cfg = ModelConfig(name="q", model="m", base_url="http://localhost:9",
                          api_key_env="FAKE_OR_KEY", price_in_per_m=0.1, price_out_per_m=0.1, **kw)
        return OpenAICompatClient(cfg, ResponseCache(tmp_path / "c.sqlite"), Budget(1.0))

    assert make().client.max_retries == 3  # the default is unchanged
    assert make(max_retries=12).client.max_retries == 12
    for name in ("e1_pilot.toml", "e1_qwen_siliconflow.toml"):
        qwen = [m for m in load_config(CONFIGS / name).models if "siliconflow" in (m.base_url or "")]
        assert qwen and all(m.max_retries and m.max_retries >= 10 for m in qwen), name


def test_local_model_needs_no_key_or_price_and_reads_its_address(tmp_path, monkeypatch):
    """A model served on your own machine: no API key, no price, address from the environment
    (so a machine's network name stays out of committed configs), and a long timeout."""
    from injbench.runner import check_prices

    monkeypatch.delenv("NO_SUCH_KEY", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setenv("QWEN_URL", "http://qwen-box.local:8000/v1")
    local = ModelConfig(name="q", model="m", api_key_env="NO_SUCH_KEY", local=True,
                        base_url_env="QWEN_URL", timeout_s=1200)
    c = OpenAICompatClient(local, ResponseCache(tmp_path / "c.sqlite"), Budget(1.0))
    assert c.client.api_key == "local"
    assert str(c.client.base_url).rstrip("/") == "http://qwen-box.local:8000/v1"
    assert c.client.timeout == 1200
    check_prices([local], budget_usd=5.0)  # free, so the budget cap does not need its price

    paid = ModelConfig(name="p", model="m", api_key_env="NO_SUCH_KEY")
    with pytest.raises(RuntimeError, match="NO_SUCH_KEY"):
        OpenAICompatClient(paid, ResponseCache(tmp_path / "c.sqlite"), Budget(1.0))
    with pytest.raises(ValueError, match="price"):
        check_prices([paid], budget_usd=5.0)
    check_prices([paid], budget_usd=None)  # no cap, nothing to count

    monkeypatch.delenv("QWEN_URL")
    with pytest.raises(RuntimeError, match="QWEN_URL"):
        OpenAICompatClient(local, ResponseCache(tmp_path / "c.sqlite"), Budget(1.0))
