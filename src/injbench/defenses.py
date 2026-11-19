"""Defense arms.

M1 ships `none` and `permissions`. Later arms plug in through the same two
hooks: a system-prompt addendum and a tool-output filter. Detectors redact
flagged segments; that on-detect action is fixed across detector arms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

OutputFilter = Callable[[str, Any], tuple[Any, bool]]


@dataclass(frozen=True)
class Defense:
    name: str
    enforce_permissions: bool = False
    system_addendum: str = ""

    def make_output_filter(self) -> OutputFilter | None:
        return None


DEFENSES: dict[str, Defense] = {
    "none": Defense("none"),
    "permissions": Defense("permissions", enforce_permissions=True),
}


def get_defense(name: str) -> Defense:
    if name not in DEFENSES:
        raise KeyError(f"unknown defense {name!r}; known: {sorted(DEFENSES)}")
    return DEFENSES[name]


# M3 detector arms (src/injbench/detect/arms.py): permissions plus a detector that redacts
# flagged tool-output segments. They subclass Defense, so this import stays below it; model
# weights load lazily when an arm builds its first filter, never at import.
from .detect.arms import DETECTOR_ARMS  # noqa: E402

DEFENSES.update(DETECTOR_ARMS)
