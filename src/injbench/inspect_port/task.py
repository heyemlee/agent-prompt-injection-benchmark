"""The injbench Inspect task: `inspect eval injbench/injbench --model <provider/model>`."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample

from .. import __version__
from .episodes import MatrixParams, Scenario, build_scenarios
from .scorer import injbench_scorer
from .solver import injbench_agent, release_episode


def scenario_sample(scenario: Scenario) -> Sample:
    """An Inspect sample for one episode. The input is the principal's request; there is no
    target, because grading checks the final state of the environment, not an answer."""
    return Sample(id=scenario.sample_id, input=scenario.task.prompt,
                  metadata=scenario.metadata())


@task
def injbench(
    templates: str | list[str] | None = None,
    instances_per_template: int = 1,
    families: str | list[str] | None = None,
    goals: str | list[str] | None = None,
    defense: str = "none",
    max_steps: int = 15,
    include_benign: bool = True,
    world_seed: int = 0,
    instance_seed: int = 0,
) -> Task:
    """Prompt-injection benchmark for a tool-using IT-ops agent (one defense arm).

    Samples are injbench episodes: every task instance once benign (if `include_benign`) and
    once per applicable attack family x goal, built by `runner.build_matrix`.

    Args:
        templates: Task templates (default: all registered), as a list or comma-separated.
        instances_per_template: Task instances per template.
        families: Attack families (default: all).
        goals: Attack goals (default: all); goals only apply to the templates they fit.
        defense: The defense arm, one of `injbench.defenses.DEFENSES`.
        max_steps: Maximum model calls per episode.
        include_benign: Also run every task instance without an attack.
        world_seed: Seed of the synthetic world.
        instance_seed: Seed for choosing task instances.
    """
    params = MatrixParams.create(templates=templates, families=families, goals=goals,
                                 defense=defense, instances_per_template=instances_per_template,
                                 include_benign=include_benign, world_seed=world_seed,
                                 instance_seed=instance_seed)
    scenarios = build_scenarios(params)
    if not scenarios:
        raise ValueError("the episode matrix is empty: select attack families and goals, or "
                         "set include_benign")
    return Task(
        dataset=MemoryDataset([scenario_sample(s) for s in scenarios],
                              name=f"injbench-{params.defense}"),
        solver=injbench_agent(max_steps=max_steps),
        scorer=injbench_scorer(),
        cleanup=release_episode,
        metadata={"injbench_version": __version__,
                  "world_fingerprint": scenarios[0].world.fingerprint(),
                  "defense": params.defense, "max_steps": max_steps,
                  "matrix": params.to_dict()},
    )
