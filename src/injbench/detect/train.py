"""Fine-tunes ModernBERT-base as a binary injection classifier on the detector dataset.

Sized for a laptop: a plain PyTorch loop (no Trainer, no accelerate), dynamic padding, fp32,
MPS when available and CPU otherwise. Training examples are the shared chunker's chunks of the
train split: a chunk of an injected segment is positive when it holds at least half of the
payload, negative when it holds none of it, and left out when it holds only a sliver. The epoch
with the lowest validation loss is kept. Seeds are explicit; MPS kernels are not bit-for-bit
deterministic, so reruns on MPS can differ slightly.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .calibration import auroc, threshold_at_fpr
from .chunking import MAX_TOKENS, chunk_label, segment_chunks
from .dataset import load_manifest, load_split
from .detectors import BASE_MODERNBERT, HFDetector, pick_device
from .paths import ensure_hf_env

LABELS = {0: "BENIGN", 1: "INJECTION"}


@dataclass
class TrainConfig:
    base_model: str = BASE_MODERNBERT
    seed: int = 0
    epochs: int = 3
    batch_size: int = 16
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_tokens: int = MAX_TOKENS
    device: str = "auto"
    class_weighted: bool = True  # inverse-frequency class weights in the loss
    max_train_rows: int | None = None  # for smoke runs
    eval_batch_size: int = 32


def chunk_examples(rows: list[dict], count_tokens: Callable[[str], int] | None,
                   max_tokens: int = MAX_TOKENS) -> list[tuple[str, int]]:
    """(chunk text, label) training examples from dataset rows."""
    out: list[tuple[str, int]] = []
    for r in rows:
        span = tuple(r["payload_span"]) if r["label"] == 1 and r.get("payload_span") else None
        for c in segment_chunks(r["text"], count_tokens, max_tokens):
            if r["label"] == 1:
                lbl = chunk_label(c, span) if span else 1
            else:
                lbl = 0
            if lbl is not None:
                out.append((c.text, lbl))
    return out


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass


def _batches(examples, batch_size, generator):
    import torch

    order = torch.randperm(len(examples), generator=generator).tolist()
    for i in range(0, len(order), batch_size):
        yield [examples[j] for j in order[i:i + batch_size]]


def _loss_on(model, tokenizer, examples, device, batch_size, max_tokens, weight=None) -> float:
    import torch

    if not examples:
        return math.nan
    total, n = 0.0, 0
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            enc = tokenizer([t for t, _ in batch], padding=True, truncation=True,
                            max_length=max_tokens, return_tensors="pt").to(device)
            labels = torch.tensor([y for _, y in batch], device=device)
            logits = model(**enc).logits.float()
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight,
                                                     reduction="sum")
            total += float(loss)
            n += len(batch)
    return total / n


def train_modernbert(data_dir: str | Path, out_dir: str | Path,
                     cfg: TrainConfig | None = None, log: Callable[[str], None] = print) -> dict:
    """Trains, keeps the best epoch by validation loss, and writes `out_dir` (weights,
    tokenizer, `train_meta.json`). Returns the metadata."""
    cfg = cfg or TrainConfig()
    ensure_hf_env()
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    from .latency import hardware_info

    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    out_dir = Path(out_dir)
    train_rows = load_split(data_dir, "train")
    val_rows = load_split(data_dir, "val")
    if cfg.max_train_rows:
        rng = random.Random(cfg.seed)
        train_rows = rng.sample(train_rows, min(cfg.max_train_rows, len(train_rows)))
    manifest = load_manifest(data_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, num_labels=2, id2label=dict(LABELS),
        label2id={v: k for k, v in LABELS.items()})
    model.to(device)

    def count_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True, truncation=False,
                             verbose=False)["input_ids"])

    train_ex = chunk_examples(train_rows, count_tokens, cfg.max_tokens)
    val_ex = chunk_examples(val_rows, count_tokens, cfg.max_tokens)
    n_pos = sum(y for _, y in train_ex)
    n_neg = len(train_ex) - n_pos
    if not n_pos or not n_neg:
        raise ValueError("training data needs both injected and benign examples")
    weight = None
    if cfg.class_weighted:
        weight = torch.tensor([len(train_ex) / (2 * n_neg), len(train_ex) / (2 * n_pos)],
                              dtype=torch.float32, device=device)

    no_decay = ("bias", "norm", "LayerNorm")
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(k in n for k in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(params, lr=cfg.lr)
    steps_per_epoch = math.ceil(len(train_ex) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    sched = get_linear_schedule_with_warmup(optim, int(cfg.warmup_ratio * total_steps),
                                            total_steps)
    gen = torch.Generator().manual_seed(cfg.seed)

    log(f"training {cfg.base_model} on {device}: {len(train_ex)} train chunks "
        f"({n_pos} injected, {n_neg} benign), {len(val_ex)} val chunks, "
        f"{cfg.epochs} epochs x {steps_per_epoch} steps")
    history, best = [], None
    t_start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, run_loss, seen = time.perf_counter(), 0.0, 0
        for batch in _batches(train_ex, cfg.batch_size, gen):
            enc = tokenizer([t for t, _ in batch], padding=True, truncation=True,
                            max_length=cfg.max_tokens, return_tensors="pt").to(device)
            labels = torch.tensor([y for _, y in batch], device=device)
            logits = model(**enc).logits.float()
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            run_loss += float(loss.detach()) * len(batch)
            seen += len(batch)
        train_secs = time.perf_counter() - t0
        val_loss = _loss_on(model, tokenizer, val_ex, device, cfg.eval_batch_size,
                            cfg.max_tokens)
        det = HFDetector.from_objects("modernbert", tokenizer, model, device,
                                      max_tokens=cfg.max_tokens,
                                      batch_size=cfg.eval_batch_size)
        scores = det.score_segments([r["text"] for r in val_rows])
        pos = [s for r, s in zip(val_rows, scores) if r["label"] == 1]
        neg = [s for r, s in zip(val_rows, scores) if r["label"] == 0]
        thr = threshold_at_fpr(neg, 0.01) if neg else math.inf
        rec = {"epoch": epoch, "train_loss": run_loss / max(seen, 1), "val_loss": val_loss,
               "val_auroc": auroc(pos, neg),
               "val_recall_at_1pct_fpr": (sum(s >= thr for s in pos) / len(pos)) if pos else None,
               "train_seconds": round(train_secs, 1)}
        history.append(rec)
        log(f"  epoch {epoch}: train loss {rec['train_loss']:.4f}, val loss {val_loss:.4f}, "
            f"val AUROC {rec['val_auroc']:.4f}, val recall@1%FPR "
            f"{rec['val_recall_at_1pct_fpr']}, {train_secs:.0f}s")
        if best is None or val_loss < best["val_loss"]:
            best = rec
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
    meta = {
        "base_model": cfg.base_model,
        "config": asdict(cfg),
        "device": device,
        "dataset_fingerprint": manifest.get("fingerprint"),
        "train_rows": len(train_rows), "val_rows": len(val_rows),
        "train_chunks": len(train_ex), "train_injected_chunks": n_pos,
        "train_benign_chunks": n_neg, "val_chunks": len(val_ex),
        "history": history, "best_epoch": best["epoch"] if best else None,
        "total_seconds": round(time.perf_counter() - t_start, 1),
        "hardware": hardware_info(device),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    log(f"best epoch {meta['best_epoch']} saved to {out_dir} ({meta['total_seconds']}s total)")
    return meta
