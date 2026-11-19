"""Detector latency: p50/p95 per call, with hardware, input length and batch size.

Two measurements:
- **Forward pass** at controlled input lengths (tokens, special tokens included) and batch
  sizes: one call scores one batch of identical chunks.
- **Per tool call**: the arm's whole output filter (segmenting, chunking, one batched forward
  pass, redaction) on real read-tool results captured by the dataset builder. Per-episode
  overhead is this times the number of read calls in an episode.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import time
from typing import Any, Callable, Sequence

from .redact import make_redaction_filter

# Benign filler for length-controlled inputs, in the world's own voice.
_FILLER = (
    "It happens a few times a day, usually after lunch. Several people near me see the same "
    "thing. I already tried restarting, no change. Is there a workaround I can use meanwhile? "
    "Open the IT self-service portal and sign in, search for the topic and follow the guided "
    "steps. If the steps fail, reply to your ticket with the error text. "
)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default), q in [0, 100]."""
    xs = sorted(values)
    if not xs:
        return float("nan")
    pos = (len(xs) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize_ms(seconds: Sequence[float]) -> dict:
    ms = [s * 1000 for s in seconds]
    return {"n": len(ms), "p50_ms": round(percentile(ms, 50), 3),
            "p95_ms": round(percentile(ms, 95), 3),
            "mean_ms": round(sum(ms) / len(ms), 3) if ms else None}


def hardware_info(device: str | None = None) -> dict[str, Any]:
    """Machine description for reports: chip and memory only (no serial numbers)."""
    info: dict[str, Any] = {"platform": platform.platform(), "machine": platform.machine(),
                            "python": platform.python_version(), "cpu_count": os.cpu_count()}
    if device:
        info["device"] = device
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True,
                                 text=True, timeout=20).stdout
            for key in ("Chip", "Memory", "Total Number of Cores"):
                m = re.search(rf"^\s*{key}:\s*(.+)$", out, re.MULTILINE)
                if m:
                    info[key.lower().replace(" ", "_")] = m.group(1).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:  # pragma: no cover
        pass
    return info


def _sized_text(detector, n_tokens: int) -> str:
    tok = detector.tokenizer
    ids = tok(_FILLER * (n_tokens // 40 + 2), add_special_tokens=False)["input_ids"]
    text = tok.decode(ids[:max(1, n_tokens - 2)])
    while detector.count_tokens(text) > n_tokens and len(text) > 1:
        text = text[:-5]
    return text


def bench_forward(detector, lengths: Sequence[int] = (32, 128, 256, 512),
                  batch_sizes: Sequence[int] = (1, 8, 32), repeats: int = 30,
                  warmup: int = 3) -> list[dict]:
    rows = []
    for n in lengths:
        text = _sized_text(detector, n)
        actual = detector.count_tokens(text)
        for bs in batch_sizes:
            batch = [text] * bs
            for _ in range(warmup):
                detector.chunk_scores(batch)
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                detector.chunk_scores(batch)
                times.append(time.perf_counter() - t0)
            rows.append({"input_tokens": actual, "batch_size": bs, **summarize_ms(times),
                         "per_item_p50_ms": round(percentile(times, 50) * 1000 / bs, 3)})
    return rows


def bench_tool_calls(score: Callable[[list[str]], Sequence[float]], threshold: float,
                     calls: Sequence[dict], warmup: int = 3, repeats: int = 1) -> dict:
    """Times the full output filter on captured read-tool results."""
    filt = make_redaction_filter(score, threshold)
    for c in calls[:warmup]:
        filt(c["tool"], c["result"])
    filt.stats.per_call_seconds.clear()
    by_tool: dict[str, list[float]] = {}
    segments = []
    for _ in range(repeats):
        for c in calls:
            before = filt.stats.segments
            t0 = time.perf_counter()
            filt(c["tool"], c["result"])
            by_tool.setdefault(c["tool"], []).append(time.perf_counter() - t0)
            segments.append(filt.stats.segments - before)
    all_times = [t for ts in by_tool.values() for t in ts]
    return {"all": summarize_ms(all_times),
            "by_tool": {t: summarize_ms(ts) for t, ts in sorted(by_tool.items())},
            "segments_per_call": {"mean": round(sum(segments) / len(segments), 2)
                                  if segments else None, "max": max(segments, default=0)}}
