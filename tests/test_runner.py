import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from injbench.cli import main
from injbench.models import Budget, BudgetExceeded, ModelConfig, OpenAICompatClient, ResponseCache
from injbench.runner import RunConfig, load_config, run_experiment

SMOKE = Path(__file__).resolve().parent.parent / "configs" / "smoke.toml"


def test_smoke_run_end_to_end(tmp_path):
    cfg = load_config(SMOKE)
    cfg.instances_per_template = 1
    cfg.cache_path = str(tmp_path / "cache.sqlite")
    run_dir = asyncio.run(run_experiment(cfg, tmp_path / "run", quiet=True))
    records = [json.loads(x) for x in (run_dir / "episodes.jsonl").read_text().splitlines()]
    assert records and all(r["status"] == "ok" for r in records)
    assert len(list((run_dir / "transcripts").iterdir())) == len(records)
    summary = json.loads((run_dir / "summary.json").read_text())
    arms = {(a["model"], a["defense"]): a["metrics"] for a in summary["arms"]}
    gold = arms[("ref-gold", "none")]
    assert gold["benign_completion"]["value"] == 1.0
    assert gold["safe_completion_under_attack"]["value"] == 1.0
    assert gold["exposure_rate"]["value"] == 1.0
    assert arms[("ref-gullible", "none")]["asr"]["value"] == 1.0
    perm = arms[("ref-gullible", "permissions")]
    assert perm["asr_out_of_permission"]["value"] == 0.0
    assert perm["asr_within_permission"]["value"] == 1.0
    assert perm["realized_violation_rate"]["value"] == 0.0
    assert perm["attempted_violation_rate"]["value"] > 0
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["spent_usd"] == 0 and manifest["world_fingerprint"]


def test_cli_dry_run(capsys):
    assert main(["run", str(SMOKE), "--dry-run"]) == 0
    assert "episodes" in capsys.readouterr().out


def test_dotenv_loader_does_not_override_shell(tmp_path, monkeypatch):
    from injbench.cli import load_dotenv

    env = tmp_path / ".env"
    env.write_text("# comment\nexport INJB_A=from_file\nINJB_B='quoted'\nnot a pair\n")
    monkeypatch.setenv("INJB_A", "from_shell")
    monkeypatch.delenv("INJB_B", raising=False)
    load_dotenv(env)
    import os
    assert os.environ["INJB_A"] == "from_shell" and os.environ["INJB_B"] == "quoted"
    monkeypatch.delenv("INJB_B")


def test_anthropic_config_loads_and_needs_a_key(tmp_path, monkeypatch):
    cfg = load_config(SMOKE.parent / "anthropic_smoke.toml")
    assert {m.kind for m in cfg.models} == {"anthropic"}
    assert all(m.temperature is None for m in cfg.models)  # never sent to current Claude models
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg.cache_path = str(tmp_path / "c.sqlite")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        asyncio.run(run_experiment(cfg, tmp_path / "r", quiet=True))


def test_api_model_needs_prices(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    cfg = RunConfig(models=[ModelConfig(name="m", model="x", api_key_env="FAKE_KEY")],
                    budget_usd=1.0, cache_path=str(tmp_path / "c.sqlite"))
    with pytest.raises(ValueError, match="price"):
        asyncio.run(run_experiment(cfg, tmp_path / "r", quiet=True))


class _FakeCompletions:
    def __init__(self):
        self.calls = 0

    async def create(self, **params):
        self.calls += 1
        call = SimpleNamespace(id="c1", function=SimpleNamespace(
            name="run_sql", arguments='{"query": "SELECT 1"}'))
        msg = SimpleNamespace(content=None, tool_calls=[call])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=0))


def test_openai_client_caches_and_charges(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    cfg = ModelConfig(name="m", model="x", api_key_env="FAKE_KEY", temperature=0.0,
                      price_in_per_m=0.5, price_out_per_m=1.0)
    budget = Budget(1.0)
    client = OpenAICompatClient(cfg, ResponseCache(tmp_path / "c.sqlite"), budget)
    fake = _FakeCompletions()
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    msgs = [{"role": "user", "content": "hi"}]
    r1 = asyncio.run(client.complete(msgs, [], attempt=0))
    r2 = asyncio.run(client.complete(msgs, [], attempt=0))
    assert r1.message["tool_calls"][0]["function"]["name"] == "run_sql"
    assert not r1.cached and r2.cached and fake.calls == 1
    assert budget.spent == pytest.approx(0.5)
    asyncio.run(client.complete(msgs, [], attempt=1))  # new attempt -> new request
    assert budget.spent == pytest.approx(1.0)
    with pytest.raises(BudgetExceeded):
        asyncio.run(client.complete(msgs, [], attempt=2))
