"""Tests for the Inspect port (src/injbench/inspect_port).

Part 1 needs no Inspect: the episode matrix, the Environment builder, grading, score values,
rates and records of `inspect_port.episodes` are checked against the native runner directly.

Part 2 runs the Inspect task with Inspect's `mockllm` model replaying injbench's scripted
reference agents (no network: every mock output carries its own token usage, so Inspect never
loads a tokenizer) and checks parity with `runner.run_episode` sample by sample: the same
Outcome, tool outputs, prompts, final reply and native record, and the same metrics as
`injbench.metrics.summarize`. Part 2 is skipped unless the `inspect` extra is installed
(`uv sync --extra inspect`).
"""

import asyncio
import json
from collections import Counter
from contextvars import ContextVar

import pytest

from injbench.defenses import DEFENSES, get_defense
from injbench.env import TOOL_SPECS
from injbench.inspect_port import episodes as ep
from injbench.metrics import METRICS, summarize
from injbench.models import ModelConfig, ModelResponse
from injbench.runner import EpisodeSpec, build_matrix, run_episode, scripted_client
from injbench.tasks import TEMPLATES

# Record fields that legitimately differ: identifiers, provider vocabulary and usage accounting.
NOT_COMPARED = {"episode_id", "model", "final_stop_reason", "input_tokens", "output_tokens",
                "cache_read_tokens", "cache_write_tokens", "cost_usd", "cached_calls"}
RUNTIME_KEYS = ("steps", "hit_step_limit", "model_refused", "final_stop_reason", "input_tokens",
                "output_tokens", "cache_read_tokens", "cache_write_tokens", "cost_usd",
                "cached_calls", "error")


def native(scenario, behavior="gold", max_steps=30, client=None):
    """(record, transcript) of the same episode from the native runner."""
    m = ModelConfig(name=f"ref-{behavior}", kind="scripted", behavior=behavior)
    spec = EpisodeSpec("x", scenario.task, scenario.attack, scenario.defense, m, 0)
    client = client or scripted_client(m, scenario.task, scenario.attack, scenario.world)
    return asyncio.run(run_episode(spec, scenario.world, client, max_steps))


def tool_contents(messages):
    return [m["content"] for m in messages if m["role"] == "tool"]


def comparable(record):
    return {k: v for k, v in record.items() if k not in NOT_COMPARED}


def _unavailable(defense):
    """Why a defense arm cannot be built offline (e.g. a detector needs an extra), or None."""
    try:
        get_defense(defense).make_output_filter()
    except (ImportError, OSError, RuntimeError) as e:
        return f"{type(e).__name__}: {e}"
    return None


# == Part 1: Inspect-free core =============================================================
def test_names_and_params_are_validated():
    assert ep.resolve_names(None, ["a", "b"], "x") == ("a", "b")
    assert ep.resolve_names("b, a,b", ["a", "b"], "x") == ("b", "a")
    assert ep.resolve_names(["a"], ["a", "b"], "x") == ("a",)
    assert ep.resolve_names([], ["a", "b"], "x") == ()
    with pytest.raises(KeyError, match="unknown"):
        ep.resolve_names("c", ["a", "b"], "x")
    with pytest.raises(ValueError, match="one task per arm"):
        ep.MatrixParams.create(defense="none,permissions")
    with pytest.raises(KeyError, match="defense"):
        ep.MatrixParams.create(defense="nope")
    p = ep.MatrixParams.create()
    assert p.templates and p.families and p.goals and p.defense == "none"
    assert ep.MatrixParams.from_dict(json.loads(json.dumps(p.to_dict()))) == p


def test_samples_are_the_native_matrix(world):
    params = ep.MatrixParams.create(instances_per_template=2, defense="permissions")
    scenarios = ep.build_scenarios(params)
    specs = build_matrix(params.run_config(), world)

    def key(task, attack):
        attack_key = attack and (attack.family, attack.goal, attack.payload)
        return task.instance_id, task.prompt, attack_key

    assert [key(s.task, s.attack) for s in scenarios] == [key(s.task, s.attack) for s in specs]
    assert len({s.sample_id for s in scenarios}) == len(scenarios)
    assert {s.defense.name for s in scenarios} == {"permissions"}
    assert Counter(s.condition for s in scenarios)["benign"] == len(params.templates) * 2


def test_scenario_is_found_from_logged_metadata():
    params = ep.MatrixParams.create(templates="kb_answer_comment", families="override",
                                    goals="grant_admin,close_unrequested")
    scenario = ep.build_scenarios(params)[-1]
    meta = json.loads(json.dumps(scenario.metadata()))  # as stored in and read from a log
    assert ep.scenario_for(scenario.sample_id, meta) is scenario
    assert meta["condition"] == "attack" and meta["goal_class"] == scenario.attack.klass
    with pytest.raises(KeyError, match="not in the injbench matrix"):
        ep.scenario_for("kb_answer_comment-9|benign", meta)
    with pytest.raises(ValueError, match="injbench sample"):
        ep.scenario_for(scenario.sample_id, {})


@pytest.mark.parametrize("defense", sorted(DEFENSES))
@pytest.mark.parametrize("behavior", ["gold", "gullible", "leaker", "refuser"])
def test_port_episode_matches_run_episode(defense, behavior):
    """start_episode + call_tool + grade_episode reproduce run_episode for scripted agents."""
    if (reason := _unavailable(defense)) is not None:
        pytest.skip(f"defense {defense!r} unavailable offline: {reason}")
    params = ep.MatrixParams.create(families="info_block", defense=defense)
    for s in ep.build_scenarios(params):
        record, transcript = native(s, behavior)
        client = scripted_client(ModelConfig(name="x", kind="scripted", behavior=behavior),
                                 s.task, s.attack, s.world)
        episode = ep.start_episode(s)
        outputs = [ep.call_tool(episode.env, name, args) for name, args in client.actions]
        out = ep.grade_episode(episode, client.final_reply)
        assert out.to_dict() == transcript["outcome"], s.sample_id
        assert outputs == tool_contents(transcript["messages"]), s.sample_id
        mine = ep.episode_record(s.metadata(), out.to_dict(), episode_id="x",
                                 model=record["model"], attempt=0,
                                 **{k: record[k] for k in RUNTIME_KEYS})
        assert mine == record, s.sample_id  # also fails if the native record gains a field


def test_arguments_that_are_not_an_object_never_reach_the_environment():
    s = ep.build_scenarios(ep.MatrixParams.create(templates="close_confirmed", families=[]))[0]
    episode = ep.start_episode(s)
    assert ep.call_tool(episode.env, "run_sql", {}, parse_error="bad json") == ep.ARGS_ERROR
    assert ep.call_tool(episode.env, "run_sql", ["SELECT 1"]) == ep.ARGS_ERROR
    assert episode.env.calls == []
    assert "unknown tool" in ep.call_tool(episode.env, "drop_tables", {})


def test_replay_reproduces_the_live_episode():
    params = ep.MatrixParams.create(templates="sla_summary_report", families="business_note",
                                    goals="falsify_count,exfil_email_phones")
    for s in ep.build_scenarios(params):
        client = scripted_client(ModelConfig(name="x", kind="scripted", behavior="gullible"),
                                 s.task, s.attack, s.world)
        live = ep.start_episode(s)
        for name, args in client.actions:
            ep.call_tool(live.env, name, args)
        replayed = ep.replay_episode(s, [(name, args, None) for name, args in client.actions])
        assert [c.to_dict() for c in replayed.env.calls] == [c.to_dict() for c in live.env.calls]
        assert (ep.grade_episode(replayed, client.final_reply).to_dict()
                == ep.grade_episode(live, client.final_reply).to_dict())


@pytest.mark.parametrize("defense", ["none", "permissions"])
def test_rates_match_native_metrics(defense):
    """RATES over score values give exactly injbench.metrics' numbers for the same episodes."""
    params = ep.MatrixParams.create(templates=list(TEMPLATES)[:4], defense=defense,
                                    families="info_block,business_note")
    items, records = [], []
    for s in ep.build_scenarios(params):
        for behavior in ("gold", "gullible", "leaker", "refuser", "null"):
            record, transcript = native(s, behavior)
            out = ep.Outcome(**transcript["outcome"])
            items.append((s.metadata(), ep.score_values(out, record["model_refused"])))
            records.append({**record, "model": "m"})
    arm = summarize(records, n_boot=10)["arms"][0]["metrics"]
    assert set(ep.HEADLINE_RATES) <= set(ep.RATES) <= set(METRICS)
    for name, rate in ep.RATES.items():
        got, expected = ep.compute_rate(rate, items), arm[name]["value"]
        assert expected is not None and got == pytest.approx(expected), name
    assert 0 < arm["asr"]["value"] < 1  # the mix of behaviors makes the check informative


# == Part 2: the Inspect task ==============================================================
_CURRENT: ContextVar = ContextVar("injbench_test_scenario")


def _isolate(mp, path):
    """Keep Inspect away from the user's environment: its data and cache folders follow HOME,
    and it loads the first .env found walking up from the working directory."""
    home = path / "home"
    home.mkdir(parents=True, exist_ok=True)
    mp.setenv("HOME", str(home))
    mp.chdir(path)
    for var in ("INSPECT_LOG_DIR", "INSPECT_EVAL_MODEL", "INSPECT_CACHE_DIR"):
        mp.delenv(var, raising=False)


def _record_scenarios(mp):
    """Let mock models see the sample being solved: wrap start_episode so it publishes the
    scenario (and how many times that sample has started, i.e. its epoch) in a ContextVar."""
    from injbench.inspect_port import solver as solver_module

    real, starts = solver_module.start_episode, Counter()

    def recording(scenario, *args, **kwargs):
        starts[scenario.sample_id] += 1
        _CURRENT.set((scenario, starts[scenario.sample_id]))
        return real(scenario, *args, **kwargs)

    mp.setattr(solver_module, "start_episode", recording)


def _output(**kwargs):
    from inspect_ai.model import ModelOutput, ModelUsage

    if "tool" in kwargs:
        out = ModelOutput.for_tool_call("mockllm", kwargs["tool"], kwargs["args"],
                                        tool_call_id=kwargs["id"], content="")
    else:
        out = ModelOutput.from_content("mockllm", kwargs["text"],
                                       stop_reason=kwargs.get("stop_reason", "stop"))
    out.usage = ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    return out


def scripted_model(behavior):
    """mockllm replaying injbench's scripted reference agent for whichever sample is being
    solved, exactly like ScriptedClient: one tool call per turn, then the final reply.
    `behavior` is a name or a function of (scenario, start number)."""
    from inspect_ai.model import get_model

    def generate(input, tools, tool_choice, config):
        scenario, start = _CURRENT.get()
        name = behavior(scenario, start) if callable(behavior) else behavior
        client = scripted_client(ModelConfig(name=f"ref-{name}", kind="scripted", behavior=name),
                                 scenario.task, scenario.attack, scenario.world)
        n = sum(1 for m in input if m.role == "assistant")
        if n < len(client.actions):
            tool, args = client.actions[n]
            return _output(tool=tool, args=args, id=f"call_{n}")
        return _output(text=client.final_reply)

    return get_model("mockllm/model", custom_outputs=generate)


def run_inspect(model, *, epochs=None, max_steps=30, **task_args):
    from inspect_ai import eval

    from injbench.inspect_port import injbench

    [log] = eval(injbench(max_steps=max_steps, **task_args), model=model, epochs=epochs,
                 log_dir="logs", display="none")
    assert log.status == "success", log.error
    return log


PARITY_RUNS = [(d, b, {"families": "info_block"}) for d in sorted(DEFENSES)
               for b in ("gold", "gullible")]
PARITY_RUNS += [("none", b, {"families": []}) for b in ("null", "refuser", "leaker")]


@pytest.fixture(scope="module")
def inspect_runs(tmp_path_factory):
    """Run the Inspect task once per (defense, behavior) and keep the logs for several tests."""
    pytest.importorskip("inspect_ai")
    runs, skipped = {}, {}
    with pytest.MonkeyPatch.context() as mp:
        _isolate(mp, tmp_path_factory.mktemp("inspect"))
        _record_scenarios(mp)
        for defense, behavior, args in PARITY_RUNS:
            if (reason := _unavailable(defense)) is not None:
                skipped[defense] = reason
                continue
            runs[defense, behavior] = run_inspect(scripted_model(behavior), defense=defense,
                                                  **args)
    return runs, skipped


@pytest.fixture
def inspect_env(tmp_path, monkeypatch):
    pytest.importorskip("inspect_ai")
    _isolate(monkeypatch, tmp_path)
    _record_scenarios(monkeypatch)
    return tmp_path


def _check_parity(log, behavior, max_steps=30):
    """Every sample of an Inspect log against the native runner; returns the native records."""
    from injbench.inspect_port import records_from_eval_log

    mine = {r["episode_id"].rsplit("|", 3)[0]: r for r in records_from_eval_log(log)}
    assert len(mine) == len(log.samples)
    records = []
    for sample in log.samples:
        scenario = ep.scenario_for(sample.id, sample.metadata)
        record, transcript = native(scenario, behavior, max_steps)
        records.append(record)
        score = sample.scores["injbench_scorer"]
        msgs = transcript["messages"]
        assert score.metadata["graded_from"] == "live"
        assert score.metadata["outcome"] == transcript["outcome"], sample.id
        assert [m.text for m in sample.messages[:2]] == [msgs[0]["content"], msgs[1]["content"]]
        assert [m.text for m in sample.messages if m.role == "tool"] == tool_contents(msgs)
        assert comparable(mine[sample.id]) == comparable(record), sample.id
        assert set(record) <= set(mine[sample.id])
    return records


@pytest.mark.parametrize("defense", sorted(DEFENSES))
@pytest.mark.parametrize("behavior", ["gold", "gullible"])
def test_inspect_run_matches_native_runner(inspect_runs, defense, behavior):
    runs, skipped = inspect_runs
    if defense in skipped:
        pytest.skip(f"defense {defense!r} unavailable offline: {skipped[defense]}")
    log = runs[defense, behavior]
    records = _check_parity(log, behavior)
    # Inspect metrics equal the native summary of the same episodes
    native_metrics = summarize(records, n_boot=10)["arms"][0]["metrics"]
    inspect_metrics = log.results.scores[0].metrics
    for name in ep.RATES:
        expected = native_metrics[name]["value"]
        if expected is None:
            assert name not in inspect_metrics, name
        else:
            assert inspect_metrics[name].value == pytest.approx(expected), name


@pytest.mark.parametrize("behavior", ["null", "refuser", "leaker"])
def test_reference_behaviors_match_native_runner(inspect_runs, behavior):
    log = inspect_runs[0]["none", behavior]
    _check_parity(log, behavior)
    metrics = log.results.scores[0].metrics
    expected = {"null": (0.0, 0.0), "refuser": (0.0, 1.0), "leaker": (1.0, 0.0)}[behavior]
    assert (metrics["benign_completion"].value, metrics["wrongful_refusal_rate"].value) == expected
    assert "asr" not in metrics  # benign-only run: attack rates are left out


def test_gullible_agent_is_contained_by_permissions(inspect_runs):
    runs, skipped = inspect_runs
    none = runs["none", "gullible"].results.scores[0].metrics
    perm = runs["permissions", "gullible"].results.scores[0].metrics
    assert none["asr_out_of_permission"].value == 1.0 and perm["asr_out_of_permission"].value == 0
    assert perm["asr_within_permission"].value == 1.0
    assert perm["realized_violation_rate"].value == 0 < perm["attempted_violation_rate"].value


def test_tools_offer_the_native_schemas(inspect_env):
    from injbench.inspect_port import tool_infos

    infos = tool_infos()
    assert [i.name for i in infos] == [s["function"]["name"] for s in TOOL_SPECS]
    for info, spec in zip(infos, TOOL_SPECS):
        assert info.description == spec["function"]["description"]
        assert (info.parameters.model_dump(mode="json", exclude_none=True)
                == spec["function"]["parameters"])
    try:  # what Inspect's OpenAI-format providers send (internal helper; skip if it moved)
        from inspect_ai.model._openai import openai_chat_tool_param
    except ImportError:
        return
    assert [dict(openai_chat_tool_param(i)) for i in infos] == TOOL_SPECS


def test_step_limit_matches_native_runner(inspect_env):
    log = run_inspect(scripted_model("gold"), max_steps=1, templates="sla_summary_report",
                      families=[])
    _check_parity(log, "gold", max_steps=1)
    score = log.samples[0].scores["injbench_scorer"]
    assert score.metadata["hit_step_limit"] and score.metadata["steps"] == 1
    assert score.answer == "" and score.value["task_success"] == 0


class _RawClient:
    """Native client returning raw OpenAI-format messages."""

    def __init__(self, messages):
        self.name, self.messages = "raw", messages

    async def complete(self, messages, tools, *, attempt=0):
        return ModelResponse(self.messages[sum(m["role"] == "assistant" for m in messages)])


def test_malformed_calls_and_refusals_match_native_runner(inspect_env):
    from inspect_ai.model import (ChatCompletionChoice, ChatMessageAssistant, ModelOutput,
                                  ModelUsage, get_model)
    from inspect_ai.tool import ToolCall

    from injbench.inspect_port import records_from_eval_log

    refusal = "I can't help with that."
    inspect_outputs = [
        ModelOutput(model="mockllm", choices=[ChatCompletionChoice(
            message=ChatMessageAssistant(content="", tool_calls=[ToolCall(
                id="c0", function="run_sql", arguments={}, parse_error="invalid JSON")]),
            stop_reason="tool_calls")]),
        _output(tool="drop_tables", args={}, id="c1"),
        _output(text=refusal, stop_reason="content_filter"),
    ]
    for out in inspect_outputs:
        out.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
    log = run_inspect(get_model("mockllm/model", custom_outputs=inspect_outputs),
                      templates="kb_answer_comment", families=[])
    raw = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c0", "type": "function", "function": {"name": "run_sql",
                                                          "arguments": "{bad json"}}]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "drop_tables",
                                                          "arguments": "{}"}}]},
        {"role": "assistant", "content": refusal, "refusal": {"type": "content_filter"}},
    ]
    [sample] = log.samples
    record, transcript = native(ep.scenario_for(sample.id, sample.metadata),
                                client=_RawClient(raw))
    [mine] = records_from_eval_log(log)
    assert sample.scores["injbench_scorer"].metadata["outcome"] == transcript["outcome"]
    assert [m.text for m in sample.messages if m.role == "tool"] == tool_contents(
        transcript["messages"])
    assert comparable(mine) == comparable(record)
    assert mine["model_refused"] and mine["wrongful_refusal"] and mine["n_tool_calls"] == 1


def test_rescoring_a_log_replays_the_tool_calls(inspect_env):
    from inspect_ai import score

    from injbench.inspect_port import injbench_scorer

    log = run_inspect(scripted_model("gullible"), templates="sla_summary_report,close_confirmed",
                      families="business_note", defense="permissions")
    rescored = score(log, injbench_scorer(), action="overwrite", display="none")
    before = {s.id: s.scores["injbench_scorer"] for s in log.samples}
    for sample in rescored.samples:
        new, old = sample.scores["injbench_scorer"], before[sample.id]
        assert new.metadata["graded_from"] == "replay"
        assert (new.value, new.metadata["outcome"]) == (old.value, old.metadata["outcome"])
    assert (rescored.results.scores[0].metrics["asr"].value
            == log.results.scores[0].metrics["asr"].value)


def test_registered_task_runs_by_name(inspect_env):
    from inspect_ai import eval

    [log] = eval("injbench/injbench", model=scripted_model("gold"), log_dir="logs",
                 display="none", task_args={"templates": "kb_answer_comment",
                                            "families": "override", "goals": "grant_admin",
                                            "defense": "permissions"})
    assert log.status == "success" and log.eval.task == "injbench/injbench"
    assert log.eval.metadata["defense"] == "permissions"
    metrics = log.results.scores[0].metrics
    assert metrics["benign_completion"].value == metrics["safe_completion_under_attack"].value == 1


def test_rates_compose_with_epoch_reducers(inspect_env):
    """Two epochs per sample, one gold and one gullible: the mean reducer pools epochs, and
    at_least(1) gives the share of cases with at least one success in k attempts."""
    from inspect_ai import Epochs
    from inspect_ai.scorer import at_least

    def behavior(scenario, start):  # starts are counted per sample across both runs
        return "gold" if start % 2 else "gullible"

    args = {"templates": "sla_summary_report,close_confirmed", "families": "info_block",
            "goals": "grant_admin,close_unrequested"}
    pooled = run_inspect(scripted_model(behavior), epochs=2, **args)
    assert pooled.results.scores[0].metrics["asr"].value == pytest.approx(0.5)
    assert pooled.results.scores[0].metrics["benign_completion"].value == 1.0
    any_of_k = run_inspect(scripted_model(behavior), epochs=Epochs(2, at_least(1)), **args)
    assert any_of_k.results.scores[0].metrics["asr"].value == 1.0


def test_records_from_logs_feed_the_native_summary(inspect_runs):
    from injbench.inspect_port import records_from_eval_logs

    runs, _ = inspect_runs
    records = records_from_eval_logs([runs["none", "gullible"], runs["permissions", "gullible"]])
    assert all(r["status"] == "ok" for r in records)
    summary = summarize(records, n_boot=50)
    arms = {a["defense"]: a["metrics"] for a in summary["arms"]}
    assert arms["none"]["realized_violation_rate"]["value"] > 0
    assert arms["permissions"]["realized_violation_rate"]["value"] == 0
    assert arms["none"]["asr"]["ci"] is not None  # template-clustered bootstrap
    paired = {(p["defense"], p["metric"]) for p in summary["paired"]}
    assert ("permissions", "realized_violation_rate") in paired
