"""Repeated attempts in the main runner: stated randomness sources, fail-fast validation, the
paraphrase matrix, and sampling keys."""

import asyncio
import json
from pathlib import Path

import pytest

from injbench.attacks import build_attack
from injbench.cli import main
from injbench.defenses import get_defense
from injbench.models import ModelConfig, ModelResponse
from injbench.runner import (EpisodeSpec, RunConfig, build_matrix, check_attempts, load_config,
                             run_episode, run_experiment)
from injbench.tasks import build_instances

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
API = dict(kind="openai_compat", model="x", api_key_env="FAKE_KEY", price_in_per_m=1.0)


def cfg(**kw):
    base = dict(templates=["sla_summary_report", "kb_answer_comment"], instances_per_template=1,
                attack_families=["info_block", "business_note"],
                attack_goals=["grant_admin", "falsify_count"], defenses=["none"],
                models=[ModelConfig(name="gold", kind="scripted")])
    base.update(kw)
    return RunConfig(**base)


def test_single_attempt_needs_no_randomness_source():
    check_attempts(cfg())
    check_attempts(cfg(models=[ModelConfig(name="claude", kind="anthropic")]))  # no temperature


def test_repeated_attempts_need_a_stated_source():
    with pytest.raises(ValueError, match="attempt_randomness"):
        check_attempts(cfg(attempts=3))
    with pytest.raises(ValueError, match="unknown attempt_randomness"):
        check_attempts(cfg(attempt_randomness="dice"))
    with pytest.raises(ValueError, match="at least 1"):
        check_attempts(cfg(attempts=0))


@pytest.mark.parametrize("temperature", [None, 0.0])
def test_temperature_source_fails_fast_without_a_temperature(temperature):
    c = cfg(attempts=3, attempt_randomness="temperature",
            models=[ModelConfig(name="api", temperature=temperature, **API)])
    with pytest.raises(ValueError, match="temperature above 0"):
        check_attempts(c)
    with pytest.raises(ValueError, match="temperature above 0"):
        build_matrix(c, None)  # validated before anything else, also on --dry-run


def test_temperature_source_rejects_scripted_and_accepts_a_sampling_model():
    with pytest.raises(ValueError, match="scripted"):
        check_attempts(cfg(attempts=2, attempt_randomness="temperature"))
    check_attempts(cfg(attempts=2, attempt_randomness="temperature",
                       models=[ModelConfig(name="api", temperature=0.7, **API)]))


def test_paraphrase_source_needs_enough_variants():
    check_attempts(cfg(attempts=64, attempt_randomness="paraphrase"))  # business_note has 64
    with pytest.raises(ValueError, match="paraphrase variants"):
        check_attempts(cfg(attempts=65, attempt_randomness="paraphrase"))


def test_load_config_fails_fast(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('[run]\nattempts = 3\n\n[[models]]\nname = "m"\nkind = "openai_compat"\n'
                    'model = "x"\nprice_in_per_m = 1.0\n')
    with pytest.raises(ValueError, match="attempt_randomness"):
        load_config(path)
    path.write_text(path.read_text().replace("attempts = 3", 'attempts = 3\n'
                                             'attempt_randomness = "temperature"'))
    with pytest.raises(ValueError, match="temperature above 0"):
        load_config(path)


def test_paraphrase_matrix(world):
    c = cfg(attempts=3, attempt_randomness="paraphrase")
    specs = build_matrix(c, world)
    assert len({s.episode_id for s in specs}) == len(specs)
    benign = [s for s in specs if s.attack is None]
    assert {s.attempt for s in benign} == {0}  # nothing varies, so benign runs once
    attack = [s for s in specs if s.attack is not None]
    assert all(s.sample_key == 0 and s.sampling_attempt == 0 for s in specs)
    by_case: dict[str, list] = {}
    for s in attack:
        by_case.setdefault(s.episode_id.rsplit("|", 1)[0], []).append(s)
    for case, runs in by_case.items():
        runs.sort(key=lambda s: s.attempt)
        assert [s.attempt for s in runs] == [0, 1, 2]
        assert [s.attack.variant for s in runs] == [0, 1, 2]
        assert len({s.attack.payload for s in runs}) == 3
        assert all(s.attack.marker in s.attack.payload for s in runs)
        canonical = build_attack(world, runs[0].task, runs[0].attack.family, runs[0].attack.goal)
        assert runs[0].attack.payload == canonical.payload  # attempt 0 reproduces earlier runs


def test_temperature_matrix(world):
    c = cfg(attempts=2, attempt_randomness="temperature",
            models=[ModelConfig(name="api", temperature=0.7, **API)])
    specs = build_matrix(c, world)
    assert {s.attempt for s in specs if s.attack is None} == {0, 1}  # independent benign samples
    assert all(s.sample_key is None and s.sampling_attempt == s.attempt for s in specs)
    pairs: dict[str, set] = {}
    for s in specs:
        if s.attack is not None:
            pairs.setdefault(s.episode_id.rsplit("|", 1)[0], set()).add(s.attack.payload)
    assert all(len(p) == 1 for p in pairs.values())  # same payload, new samples


def test_single_attempt_matrix_is_unchanged(world):
    for source in (None, "paraphrase", "temperature"):
        specs = build_matrix(cfg(attempt_randomness=source), world)
        assert all(s.attempt == 0 and s.sample_key is None for s in specs)
        assert all(s.attack is None or s.attack.variant == 0 for s in specs)


class KeyRecorder:
    name = "recorder"

    def __init__(self):
        self.keys = []

    async def complete(self, messages, tools, *, attempt=0):
        self.keys.append(attempt)
        return ModelResponse({"role": "assistant", "content": "Done."})


def test_run_episode_samples_with_the_spec_sampling_key(world):
    task = build_instances(world, ["sla_summary_report"], 1)[0]
    atk = build_attack(world, task, "info_block", "grant_admin", variant=2)
    m = ModelConfig(name="recorder", kind="scripted")
    for sample_key, expected in ((None, 2), (0, 0)):
        client = KeyRecorder()
        spec = EpisodeSpec("x", task, atk, get_defense("none"), m, 2, sample_key=sample_key)
        record, _ = asyncio.run(run_episode(spec, world, client, 5))
        assert client.keys == [expected] and record["attempt"] == 2


def test_repeated_smoke_run_reports_asr_at_k(tmp_path):
    c = load_config(CONFIGS / "attempts_smoke.toml")
    c.templates, c.attack_goals = ["sla_summary_report", "close_confirmed"], ["grant_admin",
                                                                             "close_unrequested"]
    c.cache_path = str(tmp_path / "cache.sqlite")
    run_dir = asyncio.run(run_experiment(c, tmp_path / "run", quiet=True))
    summary = json.loads((run_dir / "summary.json").read_text())
    rows = {(r["model"], r["defense"], r["scope"]): r for r in summary["asr_at_k"]["rows"]}
    assert summary["asr_at_k"]["attempts"] == 3
    assert [p["value"] for p in rows[("ref-gullible", "none", "all")]["curve"]] == [1.0] * 3
    assert [p["value"] for p in rows[("ref-gold", "none", "all")]["curve"]] == [0.0] * 3
    perm = rows[("ref-gullible", "permissions", "out_of_permission")]["curve"]
    assert [p["value"] for p in perm] == [0.0] * 3
    assert "## ASR@k" in (run_dir / "summary.md").read_text()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["config"]["attempt_randomness"] == "paraphrase"
    assert manifest["config"]["attempts"] == 3


@pytest.mark.parametrize("name", ["attempts_smoke.toml"])
def test_new_configs_dry_run(name, capsys):
    assert main(["run", str(CONFIGS / name), "--dry-run"]) == 0
    assert "episodes" in capsys.readouterr().out
