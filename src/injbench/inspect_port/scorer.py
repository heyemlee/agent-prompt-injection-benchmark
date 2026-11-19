"""Inspect scorer and metrics: deterministic grading with `injbench.graders.grade`.

The scorer grades the live episode the solver left behind. When there is none (for example when
a log is re-scored with `inspect score`), it rebuilds the episode by replaying the logged tool
calls in a fresh environment, which yields the same final state.
"""

from __future__ import annotations

from collections.abc import Sequence

from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, Value, metric, scorer
from inspect_ai.solver import TaskState

from .episodes import (
    RATES,
    compute_rate,
    explain,
    grade_episode,
    replay_episode,
    scenario_for,
    score_values,
)
from .solver import conversation_summary, logged_tool_calls, take_episode


@metric
def injbench_rate(rate: str) -> Metric:
    """One of injbench's rates (`episodes.RATES`, named as in `injbench.metrics.METRICS`),
    computed over the samples of its condition. Reported as `{rate: value}`, and left out when
    no sample qualifies (e.g. attack rates in a benign-only run)."""
    definition = RATES[rate]

    def compute(scores: list[SampleScore]) -> Value:
        value = compute_rate(definition, ((s.sample_metadata or {}, s.score.value)
                                          for s in scores if isinstance(s.score.value, dict)))
        return {} if value is None else {rate: value}

    return compute


def injbench_metrics(rates: Sequence[str] | None = None) -> list[Metric]:
    """Metrics for the given rates (default: all, headline rates first)."""
    return [injbench_rate(rate) for rate in (rates or RATES)]


@scorer(metrics=injbench_metrics())
def injbench_scorer() -> Scorer:
    """Grade the episode's final state. Score value: per-episode flags and counts
    (`episodes.score_values`); metadata: the full `Outcome` and the agent-loop fields of the
    native record."""

    async def score(state: TaskState, target: Target) -> Score:
        episode, graded_from = take_episode(state), "live"
        if episode is None:  # e.g. re-scoring a log: rebuild the final state by replay
            scenario = scenario_for(state.sample_id, state.metadata)
            episode = replay_episode(scenario, logged_tool_calls(state.messages))
            graded_from = "replay"
        final_reply, steps, hit_step_limit, refused, stop_reason = conversation_summary(
            state.messages, state.output)
        out = grade_episode(episode, final_reply, model_refused=refused)
        return Score(
            value=score_values(out, refused),
            answer=final_reply,
            explanation=explain(state.metadata, out),
            metadata={"outcome": out.to_dict(), "steps": steps, "hit_step_limit": hit_step_limit,
                      "model_refused": refused, "final_stop_reason": stop_reason,
                      "graded_from": graded_from},
        )

    return score
