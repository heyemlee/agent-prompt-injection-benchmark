"""Detectors behind one interface: a score per segment, the log-odds that it is injected.

Both detectors are Hugging Face sequence classifiers:
- **Prompt Guard 2** (`meta-llama/Llama-Prompt-Guard-2-86M`, optionally the 22M), used off the
  shelf. It is a gated model: loading needs `HF_TOKEN` for an account that accepted Meta's
  license. Without access, loading raises `DetectorUnavailable` with that explanation.
- **ModernBERT** (`answerdotai/ModernBERT-base`) fine-tuned by `train.py`, loaded from a local
  checkpoint under the artifact root.

Segments go through the shared chunker (`chunking.py`); a segment's score is the maximum over
its chunks. torch and transformers are imported only when a detector is constructed.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .calibration import Calibration
from .chunking import MAX_TOKENS, segment_chunks
from .paths import DEVICE_ENV, calibration_dir, ensure_hf_env, models_dir

BASE_MODERNBERT = "answerdotai/ModernBERT-base"


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    model: str  # hub id, or a checkpoint directory name under the artifact root's models/
    gated: bool = False
    finetuned: bool = False
    description: str = ""


DETECTOR_SPECS: dict[str, DetectorSpec] = {
    "promptguard2": DetectorSpec(
        "promptguard2", "meta-llama/Llama-Prompt-Guard-2-86M", gated=True,
        description="Llama Prompt Guard 2 86M (mDeBERTa-base), off the shelf"),
    "promptguard2-22m": DetectorSpec(
        "promptguard2-22m", "meta-llama/Llama-Prompt-Guard-2-22M", gated=True,
        description="Llama Prompt Guard 2 22M (DeBERTa-xsmall), off the shelf"),
    "modernbert": DetectorSpec(
        "modernbert", "modernbert-seed0", finetuned=True,
        description=f"{BASE_MODERNBERT} fine-tuned on the injbench detector dataset"),
}


class DetectorUnavailable(RuntimeError):
    """The detector cannot run here: missing extra, weights, access or calibration."""


def get_spec(name: str) -> DetectorSpec:
    if name not in DETECTOR_SPECS:
        raise KeyError(f"unknown detector {name!r}; known: {sorted(DETECTOR_SPECS)}")
    return DETECTOR_SPECS[name]


def resolve_model(spec: DetectorSpec, model: str | None = None,
                  root: str | Path | None = None) -> str:
    """Hub id for off-the-shelf detectors; for fine-tuned ones, the checkpoint directory (a
    name under `<root>/models/`, or an absolute path)."""
    ref = model or spec.model
    if not spec.finetuned:
        return ref
    p = Path(ref)
    return str(p if p.is_absolute() else models_dir(root) / ref)


def pick_device(pref: str | None = None) -> str:
    """`auto` (default, or `$INJBENCH_DETECT_DEVICE`) picks MPS, then CUDA, then CPU."""
    import torch

    pref = (pref or os.environ.get(DEVICE_ENV) or "auto").lower()
    if pref == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if pref == "mps" and not torch.backends.mps.is_available():
        raise DetectorUnavailable("MPS was requested but is not available here (a sandbox or VM "
                                  "without Metal access); use --device cpu")
    return pref


def label_indices(id2label: dict) -> tuple[list[int], list[int]]:
    """(positive, negative) label indices. Negative labels are named like BENIGN or SAFE; a
    binary model with generic names (LABEL_0, LABEL_1) treats index 1 as positive. With several
    positive labels (e.g. INJECTION and JAILBREAK) their probabilities are summed."""
    labels = {int(k): str(v).upper() for k, v in id2label.items()}
    neg = sorted(i for i, name in labels.items()
                 if any(w in name for w in ("BENIGN", "SAFE", "NEGATIVE", "LEGIT")))
    if not neg:
        if len(labels) != 2:
            raise ValueError(f"cannot tell the benign label among {labels}")
        neg = [0]
    pos = [i for i in sorted(labels) if i not in neg]
    if not pos:
        raise ValueError(f"no positive label among {labels}")
    return pos, neg


def _explain(name: str, model: str, err: Exception) -> str:
    first = (str(err).strip().splitlines() or [""])[0][:300]
    msg = f"{name}: cannot load {model!r} ({type(err).__name__}: {first})"
    spec = DETECTOR_SPECS.get(name)
    if spec and spec.gated:
        msg += (f". {model} is a gated model: accept Meta's license at "
                f"https://huggingface.co/{model} with your account, then set HF_TOKEN (for "
                f"example in .env; `injbench` and `injbench-detect` load it). Weights are cached "
                f"under $HF_HOME (default .cache/huggingface).")
    elif spec and spec.finetuned:
        msg += ". Train it first: `injbench-detect train`."
    return msg


class HFDetector:
    """A Hugging Face sequence classifier scored as log-odds of its non-benign labels."""

    def __init__(self, name: str, model: str, device: str = "cpu", *,
                 max_tokens: int = MAX_TOKENS, batch_size: int = 32,
                 local_files_only: bool = False):
        ensure_hf_env()
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise DetectorUnavailable(
                "detectors need the `detect` extra: `uv sync --extra detect`") from e
        self.name, self.model_id, self.device = name, model, device
        self.max_tokens, self.batch_size = max_tokens, batch_size
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=local_files_only)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model, local_files_only=local_files_only)
        except Exception as e:
            raise DetectorUnavailable(_explain(name, model, e)) from e
        self.model.to(device).eval()
        self.pos_idx, self.neg_idx = label_indices(self.model.config.id2label)
        self._torch = torch

    @classmethod
    def from_objects(cls, name: str, tokenizer, model, device: str, *,
                     max_tokens: int = MAX_TOKENS, batch_size: int = 32) -> "HFDetector":
        """Wraps an already loaded tokenizer and model (training uses this for validation)."""
        import torch

        self = cls.__new__(cls)
        self.name, self.model_id, self.device = name, getattr(model, "name_or_path", ""), device
        self.max_tokens, self.batch_size = max_tokens, batch_size
        self.tokenizer, self.model, self._torch = tokenizer, model, torch
        self.pos_idx, self.neg_idx = label_indices(model.config.id2label)
        return self

    def count_tokens(self, text: str) -> int:
        """Tokens including special tokens, as the classifier would see them."""
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False,
                                  verbose=False)["input_ids"])

    def chunk_scores(self, texts: Sequence[str]) -> list[float]:
        """Log-odds for chunks that already fit the token limit (truncation is a safety net)."""
        torch = self._torch
        out: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tokenizer(list(texts[i:i + self.batch_size]), padding=True,
                                 truncation=True, max_length=self.max_tokens,
                                 return_tensors="pt").to(self.device)
            with torch.inference_mode():
                logits = self.model(**enc).logits.double().cpu()
            pos = torch.logsumexp(logits[:, self.pos_idx], dim=1)
            neg = torch.logsumexp(logits[:, self.neg_idx], dim=1)
            out += (pos - neg).tolist()
        return out

    def chunks(self, segment: str):
        return segment_chunks(segment, self.count_tokens, self.max_tokens)

    def score_segments(self, segments: Sequence[str]) -> list[float]:
        chunk_lists = [self.chunks(s) for s in segments]
        flat = [c.text for cl in chunk_lists for c in cl]
        scores = self.chunk_scores(flat) if flat else []
        out, i = [], 0
        for cl in chunk_lists:
            out.append(max(scores[i:i + len(cl)]))
            i += len(cl)
        return out

    __call__ = score_segments

    def info(self) -> dict[str, Any]:
        return {"detector": self.name, "model": self.model_id, "device": self.device,
                "max_tokens": self.max_tokens, "batch_size": self.batch_size,
                "positive_labels": [self.model.config.id2label[i] for i in self.pos_idx],
                "parameters": sum(p.numel() for p in self.model.parameters())}


_CACHE: dict[tuple, HFDetector] = {}
_LOCK = threading.Lock()


def load_detector(name: str, model: str | None = None, *, root: str | Path | None = None,
                  device: str | None = None, local_files_only: bool = False,
                  batch_size: int = 32) -> HFDetector:
    """A loaded detector, cached per process (episodes share one copy of the weights)."""
    spec = get_spec(name)
    ref = resolve_model(spec, model, root)
    if spec.finetuned and not Path(ref).is_dir():
        raise DetectorUnavailable(f"{name}: checkpoint {ref} not found. Train it first: "
                                  f"`injbench-detect train`.")
    try:
        dev = pick_device(device)
    except ImportError as e:
        raise DetectorUnavailable(
            "detectors need the `detect` extra: `uv sync --extra detect`") from e
    key = (name, ref, dev, local_files_only, batch_size)
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = HFDetector(name, ref, dev, local_files_only=local_files_only,
                                     batch_size=batch_size)
        return _CACHE[key]


def calibration_path(tag: str, root: str | Path | None = None) -> Path:
    return calibration_dir(root) / f"{tag}.json"


def load_calibrated(tag: str, root: str | Path | None = None,
                    device: str | None = None) -> tuple[HFDetector, Calibration]:
    """The detector and frozen threshold an arm uses, from `<root>/calibration/<tag>.json`."""
    path = calibration_path(tag, root)
    if not path.exists():
        raise DetectorUnavailable(
            f"no frozen threshold for {tag!r} at {path}. Calibrate first: `injbench-detect "
            f"calibrate --detector {tag}` (needs `injbench-detect build-data`, and for "
            f"modernbert `injbench-detect train`). Set $INJBENCH_DETECT_DIR if the artifacts "
            f"live elsewhere.")
    cal = Calibration.load(path)
    return load_detector(cal.detector, cal.model, root=root, device=device), cal
