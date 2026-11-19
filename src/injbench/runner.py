"""Experiment runner: builds the episode matrix, runs it concurrently with a
budget cap and response cache, and writes JSONL records, transcripts and a
summary."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import re
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import __version__
from .agent import run_agent
from .attacks import AttackInstance, applicable, build_attack
from .defenses import Defense, get_defense
from .env import EpisodeView, Environment, diff, snapshot
from .graders import grade
from .metrics import summarize, to_markdown
from .models import (Budget, BudgetExceeded, ModelConfig, OpenAICompatClient, ResponseCache,
                     ScriptedClient)
from .tasks import TaskInstance, build_instances
from .variants import variant_count
from .world import World, build_world

MAX_LISTED = 20


@dataclass
class RunConfig:
    name: str = "run"
    world_seed: int = 0
    instance_seed: int = 0
    instances_per_template: int = 2
    templates: list[str] = field(default_factory=lambda: ["sla_summary_report"])
    attack_families: list[str] = field(default_factory=lambda: ["info_block"])
    attack_goals: list[str] = field(default_factory=lambda: ["grant_admin"])
    defenses: list[str] = field(default_factory=lambda: ["none", "permissions"])
    include_benign: bool = True
    attempts: int = 1
    # Required when attempts > 1 (PLAN §6): what makes repeated attempts differ. One of
    # RANDOMNESS_SOURCES; see docs/attacks.md.
    attempt_randomness: str | None = None
    variant_seed: int = 0  # seeds the order of paraphrase variants
    max_steps: int = 15
    concurrency: int = 4
    budget_usd: float | None = 2.0
    output_dir: str = "runs"
    cache_path: str = ".cache/responses.sqlite"
    models: list[ModelConfig] = field(default_factory=list)


RANDOMNESS_SOURCES: dict[str, str] = {
    "temperature": "independent model samples at the configured temperature (> 0); the payload "
                   "is the same in every attempt",
    "paraphrase": "attempt i carries paraphrase variant i of the payload (variant 0 is the "
                  "canonical text); identical model requests share one sample, so attempts differ "
                  "only where the payload text differs",
}


def check_attempts(cfg: RunConfig) -> None:
    """Fails fast when repeated attempts would not be independent, or their source is unstated."""
    if cfg.attempts < 1:
        raise ValueError("attempts must be at least 1")
    source = cfg.attempt_randomness
    if source is not None and source not in RANDOMNESS_SOURCES:
        raise ValueError(f"unknown attempt_randomness {source!r}; known: "
                         f"{sorted(RANDOMNESS_SOURCES)}")
    if cfg.attempts == 1:
        return
    if source is None:
        raise ValueError(f"attempts = {cfg.attempts} needs a stated randomness source: set "
                         f"attempt_randomness to one of {sorted(RANDOMNESS_SOURCES)} in [run] "
                         f"(at temperature 0, repetition means nothing)")
    if source == "temperature":
        for m in cfg.models:
            if m.kind == "scripted":
                raise ValueError(f"model {m.name!r} is scripted and never samples; use "
                                 f"attempt_randomness = \"paraphrase\"")
            if m.temperature is None or m.temperature <= 0:
                raise ValueError(
                    f"model {m.name!r}: attempt_randomness = \"temperature\" needs a temperature "
                    f"above 0 in its [[models]] entry (now {m.temperature}). For models that "
                    f"reject temperature, such as Claude 5.5, use \"paraphrase\"")
    else:
        for family in cfg.attack_families:
            n = variant_count(family)
            if n < cfg.attempts:
                raise ValueError(f"attack family {family!r} has {n} paraphrase variants, fewer "
                                 f"than attempts = {cfg.attempts}")


def load_config(path: str | Path) -> RunConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    models = [ModelConfig(**m) for m in raw.get("models", [])]
    if not models:
        raise ValueError("config needs at least one [[models]] entry")
    cfg = RunConfig(**raw.get("run", {}), models=models)
    check_attempts(cfg)
    return cfg


@dataclass
class EpisodeSpec:
    episode_id: str
    task: TaskInstance
    attack: AttackInstance | None
    defense: Defense
    model: ModelConfig
    attempt: int
    # The attempt index the model client sees; it keys the response cache, so equal keys share one
    # sample of an identical request. None means `attempt` (independent samples per attempt).
    sample_key: int | None = None

    @property
    def sampling_attempt(self) -> int:
        return self.attempt if self.sample_key is None else self.sample_key


def attempt_attacks(cfg: RunConfig, world: World, task: TaskInstance, family: str,
                    goal: str) -> list[AttackInstance]:
    """The attack of each repeated attempt: paraphrase variant i at attempt i, or the canonical
    payload every time."""
    if cfg.attempt_randomness == "paraphrase":
        return [build_attack(world, task, family, goal, variant=a, variant_seed=cfg.variant_seed)
                for a in range(cfg.attempts)]
    return [build_attack(world, task, family, goal)] * cfg.attempts


def build_matrix(cfg: RunConfig, world: World) -> list[EpisodeSpec]:
    check_attempts(cfg)
    tasks = build_instances(world, cfg.templates, cfg.instances_per_template, cfg.instance_seed)
    defenses = [get_defense(d) for d in cfg.defenses]
    # Paraphrase: attempts share samples of identical requests (sample key 0), so a benign episode
    # would only replay; it runs once.
    shared = cfg.attempts > 1 and cfg.attempt_randomness == "paraphrase"
    specs: list[EpisodeSpec] = []
    for task in tasks:
        conditions: list[tuple[str, str] | None] = [None] if cfg.include_benign else []
        conditions += [(f, g) for f in cfg.attack_families for g in cfg.attack_goals
                       if applicable(g, task)]
        for cond in conditions:
            if cond is None:
                attacks: list[AttackInstance | None] = [None] * (1 if shared else cfg.attempts)
            else:
                attacks = attempt_attacks(cfg, world, task, *cond)
            label = "benign" if cond is None else f"{cond[0]}+{cond[1]}"
            for d in defenses:
                for m in cfg.models:
                    for a, attack in enumerate(attacks):
                        specs.append(EpisodeSpec(
                            episode_id=f"{task.instance_id}|{label}|{d.name}|{m.name}|a{a}",
                            task=task, attack=attack, defense=d, model=m, attempt=a,
                            sample_key=0 if shared else None))
    return specs


def scripted_client(m: ModelConfig, task: TaskInstance, attack: AttackInstance | None,
                    world: World) -> ScriptedClient:
    actions, reply = list(task.gold_actions), task.gold_reply
    if m.behavior == "gullible" and attack is not None:
        if attack.transform_gold:
            actions = attack.transform_gold(actions)
        actions += attack.scripted_actions
    elif m.behavior == "null":
        actions, reply = [], "Done."
    elif m.behavior == "refuser":
        actions, reply = [], "I'm sorry, but I can't help with that request."
    elif m.behavior == "leaker":
        phone = world.conn.execute(
            "SELECT phone FROM users WHERE team_id IS NULL OR team_id != ? ORDER BY id LIMIT 1",
            (task.principal.team_id,)).fetchone()[0]
        reply = f"{reply} FYI, another team's on-call number is {phone}."
    elif m.behavior not in ("gold", "gullible"):
        raise ValueError(f"unknown scripted behavior {m.behavior!r}")
    return ScriptedClient(m.name, actions, reply)


def episode_seed(spec: EpisodeSpec) -> str:
    """Seed for per-episode defense parameters (the spotlighting marker).

    It depends on the task instance and the attempt only. Benign and attacked episodes of one
    instance, and arms that differ only in permissions, therefore see identical prompts until the
    defense actually changes something, and a re-run reproduces every request (cache hits). A new
    attempt gets a new marker, so an adaptive attacker cannot reuse one seen in earlier feedback.
    """
    return f"{spec.task.instance_id}|a{spec.attempt}"


async def run_episode(spec: EpisodeSpec, world: World, client, max_steps: int) -> tuple[dict, dict]:
    task, attack = spec.task, spec.attack
    defense = spec.defense.for_episode(episode_seed(spec))
    env = Environment(world, task.principal, defense.enforce_permissions,
                      attack.marker if attack else None, defense.make_output_filter(),
                      output_renderer=defense.make_output_renderer())
    if attack is not None:
        env.inject(task.vector.table, task.vector.row_id, task.vector.column, attack.payload)
    before = snapshot(env.full)
    res = await run_agent(client, env, world, task, defense, max_steps=max_steps,
                          attempt=spec.sampling_attempt)
    view = EpisodeView(env.full, task.principal, diff(before, snapshot(env.full)), env.calls,
                       res.final_reply, before)
    out = grade(task, attack, view, world, model_refused=res.refused)
    record = {
        "episode_id": spec.episode_id, "status": "ok",
        "template": task.template, "instance_id": task.instance_id,
        "condition": "benign" if attack is None else "attack",
        "family": attack.family if attack else None, "goal": attack.goal if attack else None,
        "goal_class": attack.klass if attack else None,
        "defense": defense.name, "model": spec.model.name, "attempt": spec.attempt,
        "principal_team": task.principal.team_key,
        "task_success": out.task_success, "attack_success": out.attack_success,
        "safe_completion": out.safe_completion,
        "n_realized": len(out.realized_violations), "n_attempted": len(out.attempted_violations),
        "n_unexpected_writes": len(out.unexpected_writes),
        "realized_violations": out.realized_violations[:MAX_LISTED],
        "attempted_violations": out.attempted_violations[:MAX_LISTED],
        "unexpected_writes": out.unexpected_writes[:MAX_LISTED],
        "out_of_scope_reads": out.out_of_scope_reads, "exposed": out.exposed,
        "exposed_after_filter": out.exposed_after_filter,
        "copied_marked_text": out.copied_marked_text,
        "wrongful_refusal": out.wrongful_refusal, "n_tool_calls": out.n_tool_calls,
        "steps": res.steps, "hit_step_limit": res.hit_step_limit,
        "model_refused": res.refused, "final_stop_reason": res.stop_reason,
        "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
        "cache_read_tokens": res.cache_read_tokens, "cache_write_tokens": res.cache_write_tokens,
        "cost_usd": round(res.cost_usd, 6), "cached_calls": res.cached_calls, "error": None,
    }
    transcript = {"episode_id": spec.episode_id, "prompt": task.prompt,
                  "attack_payload": attack.payload if attack else None,
                  "messages": res.messages, "calls": [c.to_dict() for c in env.calls],
                  "outcome": out.to_dict()}
    return record, transcript


API_KINDS = ("openai_compat", "anthropic")


def make_api_client(m: ModelConfig, cache: ResponseCache | None, budget: Budget):
    if m.kind == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(m, cache, budget)
    return OpenAICompatClient(m, cache, budget)


def _stub(spec: EpisodeSpec, status: str, error: str | None) -> dict:
    return {"episode_id": spec.episode_id, "status": status, "template": spec.task.template,
            "instance_id": spec.task.instance_id,
            "condition": "benign" if spec.attack is None else "attack",
            "family": spec.attack.family if spec.attack else None,
            "goal": spec.attack.goal if spec.attack else None,
            "goal_class": spec.attack.klass if spec.attack else None,
            "defense": spec.defense.name, "model": spec.model.name, "attempt": spec.attempt,
            "error": error}


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", s)


async def run_experiment(cfg: RunConfig, run_dir: str | Path | None = None,
                         quiet: bool = False) -> Path:
    world = build_world(cfg.world_seed)
    specs = build_matrix(cfg, world)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(run_dir) if run_dir else Path(cfg.output_dir) / f"{cfg.name}-{stamp}"
    (run_dir / "transcripts").mkdir(parents=True, exist_ok=True)

    budget = Budget(cfg.budget_usd)
    for m in cfg.models:
        if m.kind not in API_KINDS + ("scripted",):
            raise ValueError(f"model {m.name!r}: unknown kind {m.kind!r}")
    api_models = [m for m in cfg.models if m.kind in API_KINDS]
    for m in api_models:
        if cfg.budget_usd is not None and not (m.price_in_per_m or m.price_out_per_m):
            raise ValueError(f"model {m.name!r}: set price_in_per_m / price_out_per_m from the "
                             f"provider's price list, otherwise the budget cap cannot work")
    cache = ResponseCache(cfg.cache_path) if api_models else None
    api = {m.name: make_api_client(m, cache, budget) for m in api_models}

    sem = asyncio.Semaphore(max(1, cfg.concurrency))
    stop = asyncio.Event()
    done = 0

    async def worker(spec: EpisodeSpec) -> dict:
        nonlocal done
        async with sem:
            if stop.is_set():
                return _stub(spec, "skipped_budget", "budget reached")
            client = api.get(spec.model.name) or scripted_client(spec.model, spec.task,
                                                                 spec.attack, world)
            try:
                record, transcript = await run_episode(spec, world, client, cfg.max_steps)
            except BudgetExceeded as e:
                stop.set()
                return _stub(spec, "skipped_budget", str(e))
            except Exception as e:  # keep the run going; the record says what failed
                return _stub(spec, "error", f"{type(e).__name__}: {e}")
            path = run_dir / "transcripts" / f"{_safe_name(spec.episode_id)}.json"
            path.write_text(json.dumps(transcript, ensure_ascii=False, indent=1, default=str))
            done += 1
            if not quiet and (done % 25 == 0 or done == len(specs)):
                print(f"  {done}/{len(specs)} episodes, ${budget.spent:.4f} spent", file=sys.stderr)
            return record

    started = dt.datetime.now().isoformat(timespec="seconds")
    records = await asyncio.gather(*(worker(s) for s in specs))
    records.sort(key=lambda r: r["episode_id"])
    with open(run_dir / "episodes.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    summary = summarize(records)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "summary.md").write_text(to_markdown(summary))
    manifest = {
        "injbench_version": __version__, "started": started,
        "finished": dt.datetime.now().isoformat(timespec="seconds"),
        "world_seed": cfg.world_seed, "world_fingerprint": world.fingerprint(),
        "episodes": len(specs), "spent_usd": round(budget.spent, 6),
        "config": {**asdict(cfg), "models": [asdict(m) for m in cfg.models]},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return run_dir
