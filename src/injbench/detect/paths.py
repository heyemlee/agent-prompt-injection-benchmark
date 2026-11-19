"""Where detector artifacts live, and Hugging Face cache settings.

Everything generated here (the dataset with its attack strings, checkpoints, thresholds,
reports) goes under one git-ignored root: `artifacts/detect/` in the current directory, or
`$INJBENCH_DETECT_DIR`.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ROOT = Path("artifacts") / "detect"
ROOT_ENV = "INJBENCH_DETECT_DIR"
DEVICE_ENV = "INJBENCH_DETECT_DEVICE"


def detect_root(root: str | Path | None = None) -> Path:
    return Path(root or os.environ.get(ROOT_ENV) or DEFAULT_ROOT)


def data_dir(root: str | Path | None = None) -> Path:
    return detect_root(root) / "data"


def models_dir(root: str | Path | None = None) -> Path:
    return detect_root(root) / "models"


def calibration_dir(root: str | Path | None = None) -> Path:
    return detect_root(root) / "calibration"


def reports_dir(root: str | Path | None = None) -> Path:
    return detect_root(root) / "reports"


def ensure_hf_env() -> None:
    """Defaults for the Hugging Face libraries, set before they are imported.

    `HF_HOME` defaults to `.cache/huggingface` in the current directory (git-ignored) so that a
    sandbox that blocks `~/.cache` still works; an `HF_HOME` you set yourself wins. Telemetry is
    off. A token for gated models is read from `HF_TOKEN` (the CLIs load `.env`).
    """
    os.environ.setdefault("HF_HOME", str(Path(".cache", "huggingface").resolve()))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
