"""Inspect solver that reproduces injbench's native agent loop (`agent.run_agent`).

A custom loop rather than a generic Inspect agent, so the behavior matches the native runner:

- the same system prompt (`agent.system_prompt`, including the defense's addendum) and the
  principal's request as the only user message;
- the tools of `env.TOOL_SPECS`, offered with exactly their JSON schemas and executed by the
  episode's `Environment` (`Environment.call`), which applies the defense's permissions and
  output filter; Inspect's own tool execution (argument coercion, output truncation) is not used;
- at most `max_steps` model calls; the first reply without tool calls ends the episode, and
  its text is the final reply.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ModelOutput,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolInfo, ToolParam, ToolParams

from .. import env as injbench_env
from ..agent import system_prompt
from .episodes import Episode, call_tool, scenario_for, start_episode

# Live episodes, handed from the solver to the scorer and keyed by TaskState.uuid (unique per
# sample and epoch). The scorer removes an episode once graded; the task's cleanup hook removes
# what is left if scoring did not run.
_EPISODES: dict[str, Episode] = {}


def tool_info(spec: dict) -> ToolInfo:
    """An Inspect `ToolInfo` with exactly the JSON schema of an OpenAI-format tool spec.

    `additionalProperties` is only sent when the spec sets it (Inspect would otherwise add
    `false`), so providers receive the same schema as from the native clients.
    """
    fn = spec["function"]
    params = fn.get("parameters") or {"type": "object", "properties": {}}
    info = ToolInfo(
        name=fn["name"],
        description=fn.get("description", ""),
        parameters=ToolParams(
            properties={k: ToolParam.model_validate(v)
                        for k, v in (params.get("properties") or {}).items()},
            required=list(params.get("required") or []),
            additionalProperties=params.get("additionalProperties"),
        ),
    )
    if info.parameters.model_dump(mode="json", exclude_none=True) != params:
        raise ValueError(f"tool {fn['name']!r}: its JSON schema uses keywords that Inspect's "
                         f"ToolParams cannot represent, so the model would see a different schema")
    return info


def tool_infos() -> list[ToolInfo]:
    """Tool definitions for every tool in `env.TOOL_SPECS`, read at call time."""
    return [tool_info(spec) for spec in injbench_env.TOOL_SPECS]


@solver
def injbench_agent(max_steps: int = 15) -> Solver:
    """injbench's agent loop: one model call per step, tool calls run in the episode's
    environment, at most `max_steps` steps (the native runner's `max_steps`)."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        scenario = scenario_for(state.sample_id, state.metadata)
        episode = start_episode(scenario)  # payload injected, state snapshotted
        _EPISODES[state.uuid] = episode
        state.messages = [
            ChatMessageSystem(content=system_prompt(scenario.world, scenario.task,
                                                    scenario.defense)),
            ChatMessageUser(content=scenario.task.prompt),
        ]
        tools = tool_infos()
        model = get_model()
        for _ in range(max_steps):
            # No `cache` argument: Inspect's --cache option (GenerateConfig.cache) applies.
            state.output = await model.generate(input=state.messages, tools=tools)
            message = state.output.message
            state.messages.append(message)
            if not message.tool_calls:
                break
            for call in message.tool_calls:
                content = call_tool(episode.env, call.function, call.arguments, call.parse_error)
                state.messages.append(ChatMessageTool(content=content, tool_call_id=call.id,
                                                      function=call.function))
        return state

    return solve


def take_episode(state: TaskState) -> Episode | None:
    """The live episode of this sample, removed from the hand-over table."""
    return _EPISODES.pop(state.uuid, None)


async def release_episode(state: TaskState) -> None:
    """Task cleanup hook: drop a live episode that was not scored (frees its databases)."""
    _EPISODES.pop(state.uuid, None)


def logged_tool_calls(messages: Sequence[ChatMessage]) -> Iterator[tuple[str, Any, str | None]]:
    """(function, arguments, parse_error) of every tool call in the conversation, in order."""
    for message in messages:
        if isinstance(message, ChatMessageAssistant):
            for call in message.tool_calls or []:
                yield call.function, call.arguments, call.parse_error


def conversation_summary(messages: Sequence[ChatMessage], output: ModelOutput
                         ) -> tuple[str, int, bool, bool, str | None]:
    """(final_reply, steps, hit_step_limit, model_refused, final_stop_reason), with the native
    loop's meaning: the final reply is the text of a closing message without tool calls (empty
    if the episode ended without one); `model_refused` means that message was a refusal."""
    steps = sum(1 for m in messages if isinstance(m, ChatMessageAssistant))
    last = messages[-1] if messages else None
    replied = isinstance(last, ChatMessageAssistant) and not last.tool_calls
    stop_reason = output.choices[0].stop_reason if output.choices else None
    final_reply = last.text if replied else ""
    refused = replied and stop_reason == "content_filter"
    return final_reply, steps, not replied, refused, stop_reason
