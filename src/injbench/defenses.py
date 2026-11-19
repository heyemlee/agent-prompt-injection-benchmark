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

from dataclasses import dataclass, replace
from typing import Any, Callable

from . import spotlight

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


@dataclass(frozen=True)
class SpotlightDefense(Defense):
    """Spotlighting (Hines et al. 2024): untrusted values in tool results are delimited,
    datamarked or base64-encoded, and the system prompt says that marked text is data.

    `marker` is the datamark character or the delimiter id; empty means the mode's default.
    `for_episode` replaces it with a marker derived from the episode seed and rewrites the
    addendum to match, so the prompt and the rendered results always agree.
    """

    mode: str = "datamark"
    marker: str = ""

    def __post_init__(self):
        if self.mode not in spotlight.MODES:
            raise ValueError(f"unknown spotlight mode {self.mode!r}; known: {spotlight.MODES}")
        if not self.marker:
            object.__setattr__(self, "marker", spotlight.default_marker(self.mode))
        if not self.system_addendum:
            object.__setattr__(self, "system_addendum",
                               spotlight.addendum(self.mode, self.marker))

    def make_output_renderer(self) -> OutputRenderer | None:
        return spotlight.Spotlighter(self.mode, self.marker)

    def for_episode(self, seed: str) -> SpotlightDefense:
        marker = spotlight.choose_marker(self.mode, seed)
        return replace(self, marker=marker, system_addendum=spotlight.addendum(self.mode, marker))


DEFENSES: dict[str, Defense] = {
    "none": Defense("none"),
    "permissions": Defense("permissions", enforce_permissions=True),
}

# Spotlighting arms (M2), alone and on top of permissions.
DEFENSES.update({d.name: d for d in (
    SpotlightDefense("spotlight_delimit", mode="delimit"),
    SpotlightDefense("spotlight_datamark", mode="datamark"),
    SpotlightDefense("spotlight_base64", mode="base64"),
    SpotlightDefense("permissions+spotlight_delimit", enforce_permissions=True, mode="delimit"),
    SpotlightDefense("permissions+spotlight_datamark", enforce_permissions=True, mode="datamark"),
    SpotlightDefense("permissions+spotlight_base64", enforce_permissions=True, mode="base64"),
)})


def get_defense(name: str) -> Defense:
    if name not in DEFENSES:
        raise KeyError(f"unknown defense {name!r}; known: {sorted(DEFENSES)}")
    return DEFENSES[name]
