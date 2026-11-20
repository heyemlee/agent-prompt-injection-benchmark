"""Guards on the experiment configs that a dry run would not catch."""

import tomllib
from pathlib import Path

import pytest

from injbench.detect.dataset import DatasetConfig, assign_template_splits

CONFIGS = sorted((Path(__file__).resolve().parent.parent / "configs").glob("*.toml"))
DETECTORS = ("modernbert", "promptguard2")


def _load(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _detector_test_templates() -> set[str]:
    cfg = DatasetConfig().resolved()
    splits = assign_template_splits(cfg.templates, cfg.fractions, cfg.split_seed,
                                    cfg.val_templates, cfg.test_templates)
    return {t for t, split in splits.items() if split == "test"}


def test_detector_test_split_is_the_documented_one():
    assert _detector_test_templates() == {"access_request_handling", "chase_stale_pending",
                                          "kb_flag_outdated", "reassign_departing",
                                          "reopen_requests"}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_detector_arms_score_only_held_out_templates(path):
    """A detector arm on a template the detector was trained or calibrated on overstates it."""
    run = _load(path).get("run", {})
    if not any(d in arm for arm in run.get("defenses", []) for d in DETECTORS):
        return
    leaked = set(run.get("templates", [])) - _detector_test_templates()
    assert not leaked, f"{path.name}: detector arms on non-test templates {sorted(leaked)}"


def _local_models():
    for path in CONFIGS:
        for m in _load(path).get("models", []):
            if m.get("local"):
                yield path, m


def test_local_models_keep_their_address_out_of_the_repo_and_thinking_off():
    found = list(_local_models())
    assert found, "no config uses a local model"
    for path, m in found:
        assert "base_url" not in m and m.get("base_url_env"), path.name
        assert m.get("timeout_s", 120) >= 600, path.name
        kwargs = m["extra_params"]["extra_body"]["chat_template_kwargs"]
        assert kwargs == {"enable_thinking": False}, path.name


def test_local_qwen_configs_send_identical_requests():
    """G0, E1, E2 and the adaptive runs replay each other's episodes from the cache only if the
    model block is the same everywhere (the model string and parameters are in the cache key)."""
    blocks = {path.name: m for path, m in _local_models()}
    reference = blocks["qwen_local.toml"]
    for name, m in blocks.items():
        assert m == reference, name


def test_paid_adaptive_configs_share_one_attacker():
    """The adaptive runs are compared with each other, and the datamark and detector runs replay
    the permissions arm's attacker turns from the cache: both need one identical attacker."""
    attackers = {path.name: cfg["attacker"] for path in CONFIGS
                 if (cfg := _load(path)).get("attacker") and path.name.startswith("adaptive_")
                 and path.name not in ("adaptive_smoke.toml", "adaptive_qwen.toml")}
    assert len(attackers) == 5, sorted(attackers)
    reference = attackers["adaptive_qwen_local.toml"]
    for name, a in attackers.items():
        assert a == reference, name
    assert reference["price_in_per_m"] and reference["price_out_per_m"]
    assert reference["price_cache_read_per_m"]  # OpenAI caches prompts; bill them as cached
