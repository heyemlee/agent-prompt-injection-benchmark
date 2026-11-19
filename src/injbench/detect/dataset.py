"""Detector dataset: labeled tool-output segments from gold replays, with disjoint splits.

For every task instance (built only through the public `TEMPLATES` / `build_instances`), the
builder replays the instance's gold actions in an `Environment`: once benign, and once per
applicable attack (`FAMILIES` x `GOALS`, filtered by `applicable`, built by `build_attack`) with
the payload injected into the instance's vector record. Read-tool results are captured through
the same output-filter hook the defense arms use and cut into segments by the same
`iter_segments`, so the dataset holds exactly the units a detector arm scans at run time.

Labels: a segment from an attacked replay that contains the attack instruction is *injected*;
every segment of a benign replay is *benign*. Benign segments that are legitimate in-band
instructions from `world.py` (escalation requests, resolution confirmations) are marked as hard
negatives. Three kinds of benign text are added on top, each kept inside one split:
- hand-written hard negatives and multi-line documentation (`hard_negatives.py`), split by text;
- *composed* (format-matched) negatives: a benign body, a blank line and one of the split's
  in-band instructions, the same shape as an injected segment but without a payload.

Splits (train / val / test):
- **Templates** are disjoint: each template's instances go to one split. With fewer than three
  templates this is impossible; the builder then splits by instance and says so in the manifest.
- **Ticket sources** are disjoint: every record a segment comes from (a ticket with its notes, a
  KB article) belongs to one split. When two splits read the same record, test keeps it, then
  val, and the other split's segments from that record are dropped.
- **Attack families**: held-out families (`business_note`) appear only in test, never in train
  or val. The other ("seen") families are in every split, on disjoint templates and records.
- **Hard negatives** are in every split: tickets that carry the world's in-band instructions and
  that no template read are probed directly (`get_ticket`) and spread over the splits.

Rows are deduplicated by text within a split (`count` keeps the number of occurrences), and the
output is deterministic: the same config gives byte-identical JSONL.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .. import world as world_mod
from ..attacks import FAMILIES, GOALS, applicable, build_attack
from ..env import Environment
from ..policy import Principal
from ..tasks import TEMPLATES, TaskInstance, build_instances
from ..world import World, build_world
from .hard_negatives import hard_negative_items
from .redact import iter_segments

SPLITS = ("train", "val", "test")
HELDOUT_FAMILIES = ("business_note",)
DATA_FILES = tuple(f"{s}.jsonl" for s in SPLITS)
_CLAIM_PRIORITY = {"test": 0, "val": 1, "train": 2}
_ORIGIN_PRIORITY = {"replay": 0, "probe": 1, "augment": 2, "composed": 3}
_NAME_KEYS = frozenset({"requester", "author", "name"})


def world_hard_negatives() -> dict[str, str]:
    """Legitimate in-band instructions in the world's text pools, by text."""
    pools = (("escalation_request", "ESCALATION_REQUESTS"),
             ("resolution_confirmation", "CONFIRM_RESOLVED"))
    return {text: kind for kind, attr in pools for text in getattr(world_mod, attr, ())}


@dataclass(frozen=True)
class DatasetConfig:
    world_seeds: tuple[int, ...] = (1, 2, 3)  # the benchmark world is seed 0; keep it out
    instances_per_template: int = 6
    instance_seed: int = 0
    templates: tuple[str, ...] | None = None  # None: every template in TEMPLATES
    families: tuple[str, ...] | None = None  # None: every family in FAMILIES
    goals: tuple[str, ...] | None = None  # None: every goal in GOALS
    heldout_families: tuple[str, ...] = HELDOUT_FAMILIES
    val_templates: tuple[str, ...] | None = None  # explicit split overrides
    test_templates: tuple[str, ...] | None = None
    split_seed: int = 0
    fractions: tuple[float, float, float] = (0.5, 0.25, 0.25)
    probes: bool = True  # probe unread hard-negative tickets so every split has some
    augment: bool = True  # add the hand-written hard negatives
    compose_negatives: bool = True  # benign "host + blank line + in-band note" segments
    composed_per_host: int = 2

    def resolved(self) -> "DatasetConfig":
        return DatasetConfig(**{
            **asdict(self),
            "world_seeds": tuple(self.world_seeds),
            "templates": tuple(sorted(self.templates or TEMPLATES)),
            "families": tuple(self.families or FAMILIES),
            "goals": tuple(self.goals or GOALS),
            "heldout_families": tuple(self.heldout_families),
            "val_templates": tuple(self.val_templates) if self.val_templates else None,
            "test_templates": tuple(self.test_templates) if self.test_templates else None,
            "fractions": tuple(self.fractions),
        })


@dataclass
class Occurrence:
    """One segment as it appeared in one captured tool result."""

    text: str
    label: int  # 1 injected, 0 benign
    split: str
    origin: str  # replay | probe | augment
    template: str
    instance_id: str
    world_seed: int | None
    tool: str
    field: str  # normalized path, e.g. "notes[].body" or "rows[].body"
    field_kind: str  # title | body | name | other
    source: str  # owning record, e.g. "w1:ticket:INC-1076"
    family: str | None = None
    goal: str | None = None
    goal_class: str | None = None
    payload_span: tuple[int, int] | None = None
    hn_kind: str | None = None


@dataclass
class BuildResult:
    rows: dict[str, list[dict]]
    manifest: dict
    calls: list[dict] = field(default_factory=list)


# -- hashing helpers ------------------------------------------------------------
def _h(*parts: Any) -> int:
    blob = "\x1f".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


def _bucket(fractions: tuple[float, ...], *parts: Any) -> str:
    x = _h(*parts) / 2 ** 64
    acc = 0.0
    for split, frac in zip(SPLITS, fractions):
        acc += frac
        if x < acc:
            return split
    return SPLITS[-1]


def _spread(keys: Iterable[str], fractions: tuple[float, ...], *salt: Any) -> dict[str, str]:
    """Assigns keys to splits by hash, but the first three keys in hash order go to test, val
    and train, so every split gets at least one key when there are three or more."""
    order = sorted(set(keys), key=lambda k: _h(*salt, k))
    out = {}
    for i, k in enumerate(order):
        out[k] = ("test", "val", "train")[i] if i < 3 else _bucket(fractions, *salt, "b", k)
    return out


def assign_template_splits(templates: Iterable[str], fractions=(0.5, 0.25, 0.25), seed: int = 0,
                           val: Iterable[str] | None = None,
                           test: Iterable[str] | None = None) -> dict[str, str]:
    """Template -> split. Explicit `val` / `test` lists win; otherwise templates are ordered by a
    seeded hash and cut by `fractions`, with at least one template per split. Returns {} when
    there are fewer than three templates (no template-disjoint split exists)."""
    templates = sorted(set(templates))
    if val or test:
        val, test = set(val or ()), set(test or ())
        unknown = (val | test) - set(templates)
        if unknown:
            raise KeyError(f"split overrides name unknown templates: {sorted(unknown)}")
        if val & test:
            raise ValueError(f"templates in both val and test: {sorted(val & test)}")
        out = {t: "test" if t in test else "val" if t in val else "train" for t in templates}
        if set(out.values()) != set(SPLITS):
            raise ValueError("explicit template split leaves a split empty")
        return out
    n = len(templates)
    if n < 3:
        return {}
    n_test = max(1, round(n * fractions[2]))
    n_val = max(1, round(n * fractions[1]))
    while n - n_val - n_test < 1:
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        else:
            n_test -= 1
    order = sorted(templates, key=lambda t: _h(seed, "template", t))
    out = {t: "test" for t in order[:n_test]}
    out.update({t: "val" for t in order[n_test:n_test + n_val]})
    out.update({t: "train" for t in order[n_test + n_val:]})
    return out


# -- capture ------------------------------------------------------------------
class _Recorder:
    """An output filter that records what it sees and changes nothing."""

    def __init__(self):
        self.results: list[tuple[str, Any]] = []

    def __call__(self, tool_name: str, result: Any) -> tuple[Any, bool]:
        self.results.append((tool_name, result))
        return result, False


def replay(world: World, task: TaskInstance, attack=None) -> list[tuple[str, Any]]:
    """Runs the instance's gold actions and returns (tool, result) for every executed read call,
    exactly as a defense arm's output filter would receive them."""
    rec = _Recorder()
    env = Environment(world, task.principal, enforce_permissions=False,
                      payload_marker=attack.marker if attack else None, output_filter=rec)
    if attack is not None:
        env.inject(task.vector.table, task.vector.row_id, task.vector.column, attack.payload)
    for name, args in task.gold_actions:
        env.call(name, args)
    return rec.results


def _field(tool: str, path: tuple, result: Any) -> tuple[str, str]:
    """Normalized field name ("notes[].body", "rows[].body") and its kind."""
    cols = result.get("columns") if isinstance(result, dict) else None
    if (tool == "run_sql" and len(path) == 3 and path[0] == "rows" and cols
            and isinstance(path[2], int) and path[2] < len(cols)):
        last_name = str(cols[path[2]]).lower()  # a run_sql cell: name it by its column
        name = f"rows[].{last_name}"
    else:
        parts: list[str] = []
        last_name = ""
        for p in path:
            if isinstance(p, int):
                parts.append("[]")
            else:
                last_name = str(p).lower()
                parts.append(("." if parts else "") + str(p))
        name = "".join(parts)
    if last_name == "title":
        kind = "title"
    elif last_name == "body":
        kind = "body"
    elif last_name in _NAME_KEYS:
        kind = "name"
    else:
        kind = "other"
    return name, kind


def _source(world_seed: int, tool: str, path: tuple, result: Any) -> str | None:
    w = f"w{world_seed}"
    if not isinstance(result, dict):
        return None
    if tool == "get_ticket" and result.get("key"):
        return f"{w}:ticket:{result['key']}"
    if tool == "get_kb_article" and result.get("code"):
        return f"{w}:kb:{result['code']}"
    if tool == "search_kb" and len(path) >= 2 and path[0] == "results":
        item = result["results"][path[1]]
        if isinstance(item, dict) and item.get("code"):
            return f"{w}:kb:{item['code']}"
    if tool == "run_sql" and len(path) >= 2 and path[0] == "rows":
        cols = [str(c).lower() for c in result.get("columns") or []]
        row = result["rows"][path[1]]
        for col, kind in (("key", "ticket"), ("code", "kb")):
            if col in cols and isinstance(row, list) and cols.index(col) < len(row):
                value = row[cols.index(col)]
                if isinstance(value, str) and value:
                    return f"{w}:{kind}:{value.upper()}"
    return None


def _payload_span(text: str, attack) -> tuple[int, int]:
    i = text.rfind(attack.payload)
    if i >= 0:
        return (i, i + len(attack.payload))
    i = text.rfind(attack.marker)
    return (i, i + len(attack.marker))


def _occurrences(results, *, label_marker: str | None, attack, world_seed: int, split: str,
                 origin: str, template: str, instance_id: str,
                 hn: dict[str, str]) -> list[Occurrence]:
    out = []
    for tool, result in results:
        for path, text in iter_segments(result):
            if label_marker is not None and label_marker not in text:
                continue  # attacked replay: only the segments that carry the payload are new
            fname, fkind = _field(tool, path, result)
            source = (_source(world_seed, tool, path, result)
                      or f"w{world_seed}:unattributed:{instance_id}")
            occ = Occurrence(text=text, label=int(label_marker is not None), split=split,
                             origin=origin, template=template, instance_id=instance_id,
                             world_seed=world_seed, tool=tool, field=fname, field_kind=fkind,
                             source=source)
            if attack is not None:
                occ.family, occ.goal, occ.goal_class = attack.family, attack.goal, attack.klass
                occ.payload_span = _payload_span(text, attack)
            else:
                occ.hn_kind = hn.get(text.strip())
            out.append(occ)
    return out


# -- build ----------------------------------------------------------------------
def build_dataset(cfg: DatasetConfig | None = None,
                  worlds: dict[int, World] | None = None) -> BuildResult:
    """Builds all splits in memory. `worlds` lets tests pass prebuilt worlds by seed."""
    cfg = (cfg or DatasetConfig()).resolved()
    unknown = set(cfg.heldout_families) - set(FAMILIES)
    if unknown:
        raise KeyError(f"unknown held-out families: {sorted(unknown)}")
    hn = world_hard_negatives()
    tsplit = assign_template_splits(cfg.templates, cfg.fractions, cfg.split_seed,
                                    cfg.val_templates, cfg.test_templates)
    template_disjoint = bool(tsplit)
    occs: list[Occurrence] = []
    calls: list[dict] = []
    skipped: list[dict] = []
    unexposed: list[str] = []
    read_calls: list[int] = []
    heldout_dropped = 0

    for seed in cfg.world_seeds:
        world = (worlds or {}).get(seed) or build_world(seed)
        for template in cfg.templates:
            try:
                tasks = build_instances(world, [template], cfg.instances_per_template,
                                        cfg.instance_seed)
            except Exception as e:  # a template may not fit some world; record and go on
                skipped.append({"world_seed": seed, "template": template,
                                "error": f"{type(e).__name__}: {e}"})
                continue
            for task in tasks:
                iid = f"w{seed}:{task.instance_id}"
                split = tsplit.get(template) or _bucket(cfg.fractions, cfg.split_seed,
                                                        "instance", iid)
                results = replay(world, task)
                read_calls.append(len(results))
                common = dict(world_seed=seed, split=split, origin="replay", template=template,
                              instance_id=iid, hn=hn)
                occs += _occurrences(results, label_marker=None, attack=None, **common)
                calls += [{"split": split, "tool": t, "result": r, "injected": False,
                           "instance_id": iid} for t, r in results]
                for family in cfg.families:
                    for goal in cfg.goals:
                        if not applicable(goal, task):
                            continue
                        if family in cfg.heldout_families and split != "test":
                            heldout_dropped += 1  # never in train or val
                            continue
                        attack = build_attack(world, task, family, goal)
                        results = replay(world, task, attack)
                        found = _occurrences(results, label_marker=attack.marker, attack=attack,
                                             **common)
                        if not found:
                            unexposed.append(f"{iid}|{family}+{goal}")
                        occs += found
                        calls += [{"split": split, "tool": t, "result": r,
                                   "injected": any(attack.marker in s for _, s in
                                                   iter_segments(r)),
                                   "instance_id": iid} for t, r in results]

    # Ticket sources: one split per record; test claims first, then val, then train.
    owner: dict[str, str] = {}
    for o in occs:
        cur = owner.get(o.source)
        if cur is None or _CLAIM_PRIORITY[o.split] < _CLAIM_PRIORITY[cur]:
            owner[o.source] = o.split
    dropped = Counter(o.split for o in occs if owner[o.source] != o.split)
    occs = [o for o in occs if owner[o.source] == o.split]

    probe_count = 0
    if cfg.probes:
        for seed in cfg.world_seeds:
            world = (worlds or {}).get(seed) or build_world(seed)
            probe_occs = _probe_hard_negatives(world, seed, cfg, owner, hn)
            probe_count += len({o.instance_id for o in probe_occs})
            occs += probe_occs
    if cfg.augment:
        items = hard_negative_items()
        by_kind: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for kind, i, text in items:
            by_kind[kind].append((i, text))
        for kind, entries in by_kind.items():
            assign = _spread([t for _, t in entries], cfg.fractions, cfg.split_seed, "augment",
                             kind)
            for i, text in entries:
                occs.append(Occurrence(
                    text=text, label=0, split=assign[text], origin="augment",
                    template="augment", instance_id=f"augment:{kind}:{i}", world_seed=None,
                    tool="", field="", field_kind="body", source=f"augment:{kind}:{i}",
                    hn_kind=kind))

    if cfg.compose_negatives:
        occs += _compose_negatives(occs, cfg)

    rows = _dedup(occs, cfg.heldout_families)
    manifest = _manifest(cfg, rows, tsplit, template_disjoint, dropped, heldout_dropped,
                         skipped, unexposed, read_calls, probe_count)
    return BuildResult(rows=rows, manifest=manifest, calls=_dedup_calls(calls))


def _compose_negatives(occs: list[Occurrence], cfg: DatasetConfig) -> list[Occurrence]:
    """Format-matched negatives. An injected segment is a host text, a blank line and a payload;
    without benign segments of the same shape, a detector can learn "appended paragraph means
    injection". For each benign body in a split, this appends `composed_per_host` of that same
    split's single-line in-band instructions (hard negatives), so nothing crosses splits."""
    out: list[Occurrence] = []
    for split in SPLITS:
        hosts: dict[str, Occurrence] = {}
        kinds: dict[str, str] = {}
        for o in occs:
            if o.split != split or o.label != 0:
                continue
            if o.origin in ("replay", "probe") and o.field_kind == "body":
                hosts.setdefault(o.text, o)
            if o.hn_kind and "\n" not in o.text.strip():
                kinds.setdefault(o.text.strip(), o.hn_kind)
        if not kinds:
            continue
        for text in sorted(hosts):
            host = hosts[text]
            picks = sorted(kinds, key=lambda a: _h(cfg.split_seed, "compose", text, a))
            for note in picks[:cfg.composed_per_host]:
                out.append(Occurrence(
                    text=f"{text}\n\n{note}", label=0, split=split, origin="composed",
                    template=host.template, instance_id=host.instance_id,
                    world_seed=host.world_seed, tool=host.tool, field=host.field,
                    field_kind="body", source=host.source, hn_kind=kinds[note]))
    return out


def _probe_hard_negatives(world: World, seed: int, cfg: DatasetConfig, owner: dict[str, str],
                          hn: dict[str, str]) -> list[Occurrence]:
    """`get_ticket` on unclaimed tickets that carry a world hard negative, spread over splits."""
    conn, truth = world.conn, world.truth
    kinds = {"escalation_request": sorted(truth.escalation_ticket_ids),
             "resolution_confirmation": sorted(truth.confirmed_resolved_ids)}
    out: list[Occurrence] = []
    for kind, ticket_ids in kinds.items():
        keys = {}
        for tid in ticket_ids:
            key, team_id = conn.execute(
                "SELECT key, team_id FROM tickets WHERE id = ?", (tid,)).fetchone()
            if f"w{seed}:ticket:{key}" not in owner:
                keys[key] = team_id
        assign = _spread(keys, cfg.fractions, cfg.split_seed, "probe", seed, kind)
        for key in sorted(keys):
            lead = conn.execute("SELECT lead_user_id FROM teams WHERE id = ?",
                                (keys[key],)).fetchone()[0]
            env = Environment(world, Principal.load(conn, lead), enforce_permissions=False,
                              output_filter=(rec := _Recorder()))
            env.call("get_ticket", {"key": key})
            source = f"w{seed}:ticket:{key}"
            owner[source] = assign[key]
            out += _occurrences(rec.results, label_marker=None, attack=None, world_seed=seed,
                                split=assign[key], origin="probe", template="probe",
                                instance_id=f"w{seed}:probe:{key}", hn=hn)
    return out


def _dedup(occs: list[Occurrence], heldout: tuple[str, ...]) -> dict[str, list[dict]]:
    groups: dict[tuple[str, str], list[Occurrence]] = defaultdict(list)
    for o in occs:
        groups[(o.split, o.text)].append(o)
    rows: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for (split, text), group in groups.items():
        labels = {o.label for o in group}
        if len(labels) != 1:
            raise ValueError(f"segment labeled both injected and benign in {split}: {text[:80]!r}")
        group.sort(key=lambda o: (_ORIGIN_PRIORITY[o.origin], o.instance_id, o.tool, o.field))
        first = group[0]
        attacked = [o for o in group if o.family]
        a = attacked[0] if attacked else None
        hn_kinds = sorted({o.hn_kind for o in group if o.hn_kind})
        rows[split].append({
            "id": hashlib.sha256(f"{split}\x00{text}".encode()).hexdigest()[:16],
            "split": split,
            "text": text,
            "label": first.label,
            "origin": first.origin,
            "family": a.family if a else None,
            "goal": a.goal if a else None,
            "goal_class": a.goal_class if a else None,
            "family_seen": (a.family not in heldout) if a else None,
            "payload_span": list(a.payload_span) if a and a.payload_span else None,
            "hard_negative": bool(hn_kinds),
            "hn_kind": hn_kinds[0] if hn_kinds else None,
            "hn_origin": ({"augment": "augment", "composed": "composed"}.get(first.origin, "world")
                          if hn_kinds else None),
            "templates": sorted({o.template for o in group}),
            "instance_ids": sorted({o.instance_id for o in group}),
            "world_seeds": sorted({o.world_seed for o in group if o.world_seed is not None}),
            "tools": sorted({o.tool for o in group if o.tool}),
            "fields": sorted({o.field for o in group if o.field}),
            "field_kind": first.field_kind,
            "sources": sorted({o.source for o in group}),
            "count": len(group),
            "n_chars": len(text),
            "seen_in_train": False,
        })
    train_texts = {r["text"] for r in rows["train"]}
    for split in ("val", "test"):
        for r in rows[split]:
            r["seen_in_train"] = r["text"] in train_texts
    for split in SPLITS:
        rows[split].sort(key=lambda r: r["id"])
    return rows


def _dedup_calls(calls: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in calls:
        blob = json.dumps(c["result"], sort_keys=True, ensure_ascii=False, default=str)
        key = (c["tool"], blob)
        if key not in seen:
            seen.add(key)
            out.append({"split": c["split"], "tool": c["tool"], "injected": c["injected"],
                        "instance_id": c["instance_id"], "result": c["result"]})
    out.sort(key=lambda c: (c["instance_id"], c["tool"],
                            json.dumps(c["result"], sort_keys=True, default=str)))
    return out


# -- manifest and stats -------------------------------------------------------------------
def dataset_stats(rows: dict[str, list[dict]]) -> dict:
    out = {}
    for split, rs in rows.items():
        inj = [r for r in rs if r["label"] == 1]
        ben = [r for r in rs if r["label"] == 0]
        out[split] = {
            "rows": len(rs), "injected": len(inj), "benign": len(ben),
            "occurrences": sum(r["count"] for r in rs),
            "hard_negatives": dict(sorted(Counter(
                f"{r['hn_origin']}:{r['hn_kind']}" for r in ben if r["hard_negative"]).items())),
            "benign_by_origin": dict(sorted(Counter(r["origin"] for r in ben).items())),
            "benign_by_field_kind": dict(sorted(Counter(r["field_kind"] for r in ben).items())),
            "injected_by_family": dict(sorted(Counter(r["family"] for r in inj).items())),
            "injected_by_goal": dict(sorted(Counter(r["goal"] for r in inj).items())),
            "injected_seen_family": sum(bool(r["family_seen"]) for r in inj),
            "injected_unseen_family": sum(not r["family_seen"] for r in inj),
            "templates": sorted({t for r in rs for t in r["templates"]}),
            "sources": len({s for r in rs for s in r["sources"]}),
            "benign_texts_also_in_train": (sum(r["seen_in_train"] for r in ben)
                                           if split != "train" else None),
            "max_chars": max((r["n_chars"] for r in rs), default=0),
        }
    return out


def _fingerprint(rows: dict[str, list[dict]]) -> str:
    h = hashlib.sha256()
    for split in SPLITS:
        h.update(split.encode())
        for r in rows[split]:
            h.update(_dump(r).encode())
    return h.hexdigest()[:16]


def _manifest(cfg, rows, tsplit, template_disjoint, dropped, heldout_dropped, skipped,
              unexposed, read_calls, probe_count) -> dict:
    sources = {s: {src for r in rows[s] for src in r["sources"]} for s in SPLITS}
    overlap = {f"{a}&{b}": len(sources[a] & sources[b])
               for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]}
    return {
        "fingerprint": _fingerprint(rows),
        "config": asdict(cfg),
        "template_splits": dict(sorted(tsplit.items())),
        "template_disjoint": template_disjoint,
        "preliminary": len(cfg.templates) < 20,  # PLAN targets 20+ templates (M2)
        "seen_families": [f for f in cfg.families if f not in cfg.heldout_families],
        "heldout_families": list(cfg.heldout_families),
        "stats": dataset_stats(rows),
        "source_overlap_between_splits": overlap,
        "dropped_for_source_disjointness": dict(sorted(dropped.items())),
        "heldout_attacks_skipped_outside_test": heldout_dropped,
        "hard_negative_probe_tickets": probe_count,
        "skipped_templates": skipped,
        "attacks_without_exposed_segment": unexposed,
        "read_calls_per_gold_episode": {
            "mean": (sum(read_calls) / len(read_calls)) if read_calls else 0.0,
            "max": max(read_calls, default=0),
        },
    }


def _dump(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def write_dataset(result: BuildResult, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in result.rows[split]:
                f.write(_dump(r) + "\n")
    with open(out / "calls.jsonl", "w", encoding="utf-8") as f:
        for c in result.calls:
            f.write(_dump(c) + "\n")
    (out / "manifest.json").write_text(json.dumps(result.manifest, indent=1, sort_keys=True)
                                       + "\n")
    return out


def load_split(data: str | Path, split: str) -> list[dict]:
    path = Path(data) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `injbench-detect build-data` first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_manifest(data: str | Path) -> dict:
    path = Path(data) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `injbench-detect build-data` first")
    return json.loads(path.read_text())


def load_calls(data: str | Path) -> list[dict]:
    path = Path(data) / "calls.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
