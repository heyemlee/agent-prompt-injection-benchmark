"""Native injbench records from Inspect eval logs.

`records_from_eval_log` turns a log of the injbench task into the records the native runner
writes to `episodes.jsonl`. `injbench.metrics.summarize` and `to_markdown` then report
template-clustered bootstrap CIs, and paired differences between defense arms when the logs of
several arms (same model) are combined.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from inspect_ai.log import EvalLog, EvalSample, read_eval_log

from .episodes import MATRIX_KEY, episode_record

SCORER = "injbench_scorer"


def _usage(sample: EvalSample) -> dict:
    totals = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
              "cache_write_tokens": 0}
    cost: float | None = None
    for usage in (sample.model_usage or {}).values():
        totals["input_tokens"] += usage.input_tokens or 0
        totals["output_tokens"] += usage.output_tokens or 0
        totals["cache_read_tokens"] += usage.input_tokens_cache_read or 0
        totals["cache_write_tokens"] += usage.input_tokens_cache_write or 0
        if usage.total_cost is not None:
            cost = (cost or 0.0) + usage.total_cost
    # Inspect does not report per-sample cache hits, and its cost needs model cost data.
    return {**totals, "cost_usd": cost, "cached_calls": None}


def records_from_eval_log(log: EvalLog | str | os.PathLike, scorer: str = SCORER) -> list[dict]:
    """One native record per injbench sample and epoch (epoch k is attempt k-1).

    Samples that errored or were not scored become `status: "error"` records, as in the
    native runner. Usage fields follow Inspect's accounting, which differs by provider.
    """
    if not isinstance(log, EvalLog):
        log = read_eval_log(os.fspath(log))
    model = log.eval.model
    records = []
    for sample in log.samples or []:
        meta = sample.metadata or {}
        if MATRIX_KEY not in meta:
            continue
        attempt = (sample.epoch or 1) - 1
        episode_id = f"{sample.id}|{meta['defense']}|{model}|a{attempt}"
        score = (sample.scores or {}).get(scorer)
        details = score.metadata if score is not None else None
        if not details or "outcome" not in details:
            records.append({
                "episode_id": episode_id, "status": "error", "template": meta["template"],
                "instance_id": meta["instance_id"], "condition": meta["condition"],
                "family": meta["family"], "goal": meta["goal"], "goal_class": meta["goal_class"],
                "defense": meta["defense"], "model": model, "attempt": attempt,
                "error": sample.error.message if sample.error else "not scored"})
            continue
        records.append(episode_record(
            meta, details["outcome"], episode_id=episode_id, model=model, attempt=attempt,
            steps=details["steps"], hit_step_limit=details["hit_step_limit"],
            model_refused=details["model_refused"], final_stop_reason=details["final_stop_reason"],
            **_usage(sample)))
    records.sort(key=lambda r: r["episode_id"])
    return records


def records_from_eval_logs(logs: Iterable[EvalLog | str | os.PathLike],
                           scorer: str = SCORER) -> list[dict]:
    """Records of several logs, e.g. one per defense arm, ready for `metrics.summarize`."""
    return [r for log in logs for r in records_from_eval_log(log, scorer)]
