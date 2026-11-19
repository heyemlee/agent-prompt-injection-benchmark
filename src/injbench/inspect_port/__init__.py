"""injbench on the UK AISI Inspect framework (`inspect_ai`).

Install the optional extra (`uv sync --extra inspect`), then run, for example::

    inspect eval injbench/injbench --model anthropic/claude-sonnet-5-5 -T defense=permissions

The Inspect task uses injbench's own episodes, tools, defenses and deterministic graders; see
docs/inspect.md. Importing this package does not import `inspect_ai`: the Inspect-specific names
below are loaded on first use, and `episodes` (the Inspect-free core) works without the extra.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "injbench": ".task",
    "scenario_sample": ".task",
    "injbench_agent": ".solver",
    "tool_infos": ".solver",
    "injbench_scorer": ".scorer",
    "injbench_rate": ".scorer",
    "injbench_metrics": ".scorer",
    "records_from_eval_log": ".logs",
    "records_from_eval_logs": ".logs",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        return getattr(import_module(module, __name__), name)
    except ModuleNotFoundError as e:
        if e.name == "inspect_ai" or (e.name or "").startswith("inspect_ai."):
            raise ImportError(f"injbench.inspect_port.{name} needs Inspect: install the "
                              f"'inspect' extra (uv sync --extra inspect)") from e
        raise
