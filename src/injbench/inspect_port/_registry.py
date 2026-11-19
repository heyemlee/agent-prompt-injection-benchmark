"""Inspect entry point (`[project.entry-points.inspect_ai]` in pyproject.toml).

Inspect imports this module when it meets a name in the `injbench/` namespace, which registers
the task (`inspect eval injbench/injbench`), the solver, the scorer and the metrics, so logs can
also be re-scored and their metrics recomputed by name.
"""

# ruff: noqa: F401
from .scorer import injbench_rate, injbench_scorer
from .solver import injbench_agent
from .task import injbench
