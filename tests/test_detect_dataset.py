"""The detector dataset: determinism, labels, disjoint splits, held-out family, hard negatives."""

import pytest

from injbench.attacks import FAMILIES
from injbench.detect.dataset import (SPLITS, DatasetConfig, assign_template_splits,
                                     build_dataset, load_split, replay, write_dataset)
from injbench.policy import WRITE_TOOLS
from injbench.tasks import TEMPLATES, build_instances

# The static text before the instruction in each family's payload, e.g. "[SYSTEM NOTICE ".
PREFIX = {name: tmpl.split("{")[0] for name, tmpl in FAMILIES.items()}
PSEUDO = {"probe", "augment"}
# Train templates run in world 0, val templates in world 4, test templates in world 5.
CFG = DatasetConfig(world_seeds=(0,), val_world_seeds=(4,), test_world_seeds=(5,),
                    instances_per_template=2)


@pytest.fixture(scope="module")
def built(world):
    return build_dataset(CFG, worlds={0: world})


def test_build_is_deterministic(world, tmp_path, built):
    again = build_dataset(CFG, worlds={0: world})
    assert again.manifest["fingerprint"] == built.manifest["fingerprint"]
    assert again.rows == built.rows
    a, b = tmp_path / "a", tmp_path / "b"
    write_dataset(built, a)
    write_dataset(again, b)
    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "manifest.json", "calls.jsonl"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    assert load_split(a, "test") == built.rows["test"]


def test_labels(built):
    for split, rows in built.rows.items():
        assert rows, split
        for r in rows:
            if r["label"] == 1:
                s, e = r["payload_span"]
                assert 0 <= s < e <= len(r["text"])
                assert r["text"][s:e].startswith(PREFIX[r["family"]]), r["id"]
                assert r["goal"] and r["goal_class"] and r["origin"] == "replay"
                assert not r["hard_negative"]
            else:
                assert r["family"] is None and r["payload_span"] is None
                assert not any(p in r["text"] for p in PREFIX.values()), r["text"]


def test_rows_record_provenance(built):
    worlds = built.manifest["split_world_seeds"]
    assert worlds == {"train": [0], "val": [4], "test": [5]}
    for split, rows in built.rows.items():
        for r in rows:
            assert r["templates"] and r["instance_ids"] and r["sources"] and r["count"] >= 1
            if r["origin"] != "augment":
                assert r["tools"] and r["fields"] and r["world_seeds"] == worlds[split]
                assert not set(r["tools"]) & WRITE_TOOLS


def test_split_worlds_keep_every_template(built):
    """Separate worlds per split make records disjoint by construction: nothing is dropped,
    and every template contributes injected segments to its split."""
    assert not any(built.manifest["dropped_for_source_disjointness"].values())
    injected = {t for rows in built.rows.values() for r in rows if r["label"] == 1
                for t in r["templates"]}
    assert injected == set(TEMPLATES)


def test_shared_worlds_fall_back_to_claiming_records(world):
    shared = build_dataset(DatasetConfig(world_seeds=(0,), val_world_seeds=None,
                                         test_world_seeds=None, instances_per_template=1,
                                         probes=False, augment=False), worlds={0: world})
    m = shared.manifest
    assert m["split_world_seeds"] == {"train": [0], "val": [0], "test": [0]}
    assert not any(m["source_overlap_between_splits"].values())
    assert sum(m["dropped_for_source_disjointness"].values()) > 0


def test_split_world_seeds_must_not_overlap():
    with pytest.raises(ValueError, match="overlap"):
        DatasetConfig(world_seeds=(1, 2), val_world_seeds=(2,)).split_world_seeds(True)
    with pytest.raises(ValueError, match="overlap"):
        DatasetConfig(val_world_seeds=(4,), test_world_seeds=(4,)).split_world_seeds(True)
    with pytest.raises(ValueError, match="empty"):
        DatasetConfig(test_world_seeds=()).split_world_seeds(True)
    # Instance-level splits (fewer than three templates) keep every split in the train worlds.
    assert DatasetConfig().split_world_seeds(False) == {s: (1, 2, 3) for s in SPLITS}


def test_splits_are_disjoint(built):
    rows = built.rows
    templates = {s: {t for r in rows[s] for t in r["templates"]} - PSEUDO for s in SPLITS}
    sources = {s: {x for r in rows[s] for x in r["sources"]} for s in SPLITS}
    instances = {s: {x for r in rows[s] for x in r["instance_ids"]} for s in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            assert not templates[a] & templates[b], (a, b)
            assert not sources[a] & sources[b], (a, b)
            assert not instances[a] & instances[b], (a, b)
    assert built.manifest["template_disjoint"]
    assert set(built.manifest["source_overlap_between_splits"].values()) == {0}
    assert all(templates[s] for s in SPLITS)


def test_business_note_is_never_in_train_or_val(built):
    for split in ("train", "val"):
        assert not any(r["family"] == "business_note" for r in built.rows[split])
        assert all(r["family_seen"] for r in built.rows[split] if r["label"] == 1)
    test_inj = [r for r in built.rows["test"] if r["label"] == 1]
    assert any(r["family"] == "business_note" and r["family_seen"] is False for r in test_inj)
    assert any(r["family_seen"] for r in test_inj)  # seen families are tested too
    assert built.manifest["heldout_families"] == ["business_note"]


def test_hard_negatives_in_every_split(built):
    for split, rows in built.rows.items():
        kinds = {(r["hn_origin"], r["hn_kind"]) for r in rows if r["hard_negative"]}
        assert ("world", "escalation_request") in kinds, split
        assert ("world", "resolution_confirmation") in kinds, split
        assert any(origin == "augment" for origin, _ in kinds), split
        assert all(r["label"] == 0 for r in rows if r["hard_negative"])
    augmented = {s: {r["text"] for r in built.rows[s] if r["hn_origin"] == "augment"}
                 for s in SPLITS}
    assert not augmented["train"] & augmented["test"] and not augmented["val"] & augmented["test"]


def test_composed_negatives_stay_inside_their_split(built):
    for split, rows in built.rows.items():
        composed = [r for r in rows if r["origin"] == "composed"]
        assert composed, split
        bodies = {r["text"] for r in rows if r["origin"] in ("replay", "probe")
                  and r["field_kind"] == "body"}
        notes = {r["text"].strip() for r in rows if r["hard_negative"]
                 and r["origin"] != "composed"}
        for r in composed:
            host, note = r["text"].rsplit("\n\n", 1)
            assert r["label"] == 0 and r["hn_origin"] == "composed"
            assert host in bodies and note in notes, (split, r["text"][:60])
    docs = [r for r in built.rows["train"] if r["hn_kind"] == "documentation"]
    assert docs and all("\n" in r["text"] for r in docs)


def test_replay_captures_read_tools_only(world):
    task = build_instances(world, ["kb_answer_comment"], 1)[0]
    tools = [name for name, _ in replay(world, task)]
    assert tools == ["search_kb", "get_kb_article"]


def test_template_split_assignment():
    assert assign_template_splits(["a"]) == {} and assign_template_splits(["a", "b"]) == {}
    three = assign_template_splits(["a", "b", "c"])
    assert sorted(three.values()) == ["test", "train", "val"]
    four = assign_template_splits(["a", "b", "c", "d"])
    assert sorted(four.values()) == ["test", "train", "train", "val"]
    names = [f"t{i}" for i in range(20)]
    twenty = assign_template_splits(names)
    counts = {s: list(twenty.values()).count(s) for s in SPLITS}
    assert counts == {"train": 10, "val": 5, "test": 5}
    assert assign_template_splits(reversed(names)) == twenty  # order of input does not matter
    assert assign_template_splits(names, seed=1) != twenty
    explicit = assign_template_splits(["a", "b", "c", "d"], val=["b"], test=["c"])
    assert explicit == {"a": "train", "b": "val", "c": "test", "d": "train"}
    with pytest.raises(KeyError):
        assign_template_splits(["a", "b", "c"], test=["zzz"])
    with pytest.raises(ValueError):
        assign_template_splits(["a", "b", "c"], val=["a"], test=["a"])
    with pytest.raises(ValueError):
        assign_template_splits(["a", "b", "c"], val=["a", "b", "c"])


def test_new_templates_flow_in_without_code_changes(world):
    """M2's templates arrive through the public registry; the split stays template-disjoint."""
    assert len(TEMPLATES) >= 20
    res = build_dataset(DatasetConfig(world_seeds=(0,), instances_per_template=1,
                                      probes=False, augment=False), worlds={0: world})
    splits = res.manifest["template_splits"]
    assert set(splits) == set(TEMPLATES) and res.manifest["template_disjoint"]
    counts = {s: list(splits.values()).count(s) for s in SPLITS}
    assert counts["val"] >= 5 and counts["test"] >= 5
    assert counts["train"] >= len(TEMPLATES) // 2
    for split, rows in res.rows.items():
        assert {t for r in rows for t in r["templates"]} <= {
            t for t, s in splits.items() if s == split}


def test_fewer_than_three_templates_fall_back_to_instance_splits(world):
    res = build_dataset(DatasetConfig(world_seeds=(0,), instances_per_template=6,
                                      templates=("close_confirmed", "kb_answer_comment")),
                        worlds={0: world})
    assert not res.manifest["template_disjoint"] and res.manifest["template_splits"] == {}
    sources = {s: {x for r in res.rows[s] for x in r["sources"]} for s in SPLITS}
    assert not (sources["train"] & sources["test"]) and not (sources["val"] & sources["test"])
    assert not any(r["family"] == "business_note"
                   for s in ("train", "val") for r in res.rows[s])
