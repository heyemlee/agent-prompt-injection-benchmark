"""M3 injection detectors: dataset, chunking, detectors, training, calibration, latency, arms.

Importing this package (or `injbench`) never imports torch or transformers. The modules that
need them (`detectors`, `train`, `latency`) import them inside functions, and the defense arms
load model weights lazily, the first time an arm builds its output filter.

Layout:
  paths.py           artifact directory and Hugging Face cache settings
  redact.py          tool-output segments and the fixed on-detect action (redaction)
  chunking.py        the one chunker every detector shares (at most 512 tokens per chunk)
  hard_negatives.py  hand-written benign in-band instructions (augmentation, text-disjoint splits)
  dataset.py         dataset builder: gold replays with and without payloads, splits, JSONL
  calibration.py     threshold at a target FPR, rates with Wilson intervals, AUROC
  detectors.py       Prompt Guard 2 and fine-tuned ModernBERT behind one scoring interface
  train.py           ModernBERT fine-tune
  evaluate.py        offline evaluation, error analysis, end-to-end proxy with scripted agents
  latency.py         p50/p95 latency per call
  arms.py            `modernbert`, `promptguard2`, and each on top of `permissions`
  cli.py             `injbench-detect`
"""

# Import order matters: injbench.defenses registers the detector arms by importing .arms at its
# end, and .arms subclasses injbench.defenses.Defense. Loading defenses first makes every import
# order work (`import injbench.detect.arms` as well as `import injbench.defenses`).
from .. import defenses as _defenses  # noqa: F401
