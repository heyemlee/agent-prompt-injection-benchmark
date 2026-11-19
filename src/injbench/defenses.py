"""Defense arms.

M1 ships `none` and `permissions`. Later arms plug in through three hooks:

- a system-prompt addendum (`system_addendum`);
- a tool-output filter (`make_output_filter`) that removes content. Detectors redact flagged
  segments; that on-detect action is fixed across detector arms;
- a tool-output renderer (`make_output_renderer`) that changes presentation only, such as
  spotlighting. It runs after exposure tracking and after the filter, so it cannot hide a payload
  from the exposure metrics (see `Environment.call`).

`for_episode(seed)` returns the arm as used in one episode. Arms with per-episode parameters
(the spotlighting marker) derive them deterministically from the seed; the others return
themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

OutputFilter = Callable[[str, Any], tuple[Any, bool]]
OutputRenderer = Callable[[str, Any], Any]


@dataclass(frozen=True)
class Defense:
    name: str
    enforce_permissions: bool = False
    system_addendum: str = ""

    def make_output_filter(self) -> OutputFilter | None:
        return None

    def make_output_renderer(self) -> OutputRenderer | None:
        """`fn(tool_name, result) -> new_result` applied to executed read tools' results after
        the output filter. Presentation only: it must not remove content, and it returns a new
        object instead of mutating its input."""
        return None

    def for_episode(self, seed: str) -> Defense:
        """The arm as used in one episode (see `runner.episode_seed`)."""
        return self


DEFENSES: dict[str, Defense] = {
    "none": Defense("none"),
    "permissions": Defense("permissions", enforce_permissions=True),
}


def get_defense(name: str) -> Defense:
    if name not in DEFENSES:
        raise KeyError(f"unknown defense {name!r}; known: {sorted(DEFENSES)}")
    return DEFENSES[name]
