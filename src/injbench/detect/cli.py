"""`injbench-detect`: build the detector dataset, train, calibrate, evaluate, time the arms.

Every artifact goes under `artifacts/detect/` (or `--root`, or `$INJBENCH_DETECT_DIR`), which is
git-ignored: the dataset contains generated attack strings. Commands that need torch require the
`detect` extra (`uv sync --extra detect`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .paths import (DEVICE_ENV, ROOT_ENV, data_dir, detect_root, ensure_hf_env, models_dir,
                    reports_dir)

EXIT_UNAVAILABLE = 3


def _pct(d: dict | None) -> str:
    if not d or not d.get("n"):
        return "n/a"
    lo, hi = d["ci95"]
    return f"{d['rate']:.3f} ({d['k']}/{d['n']}) [{lo:.3f}, {hi:.3f}]"


def _fmt(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.4f}"


def _unavailable(e: Exception) -> int:
    print(f"skipped: {e}", file=sys.stderr)
    return EXIT_UNAVAILABLE


# -- commands ---------------------------------------------------------------------------
def cmd_build_data(args) -> int:
    from .dataset import DatasetConfig, build_dataset, write_dataset

    cfg = DatasetConfig(
        world_seeds=tuple(args.world_seeds), instances_per_template=args.instances,
        instance_seed=args.instance_seed, split_seed=args.split_seed,
        templates=tuple(args.templates) if args.templates else None,
        val_templates=tuple(args.val_templates) if args.val_templates else None,
        test_templates=tuple(args.test_templates) if args.test_templates else None,
        probes=not args.no_probes, augment=not args.no_augment,
        compose_negatives=not args.no_compose)
    result = build_dataset(cfg)
    out = write_dataset(result, data_dir(args.root))
    _print_stats(result.manifest)
    print(f"wrote {out} (fingerprint {result.manifest['fingerprint']})")
    return 0


def _print_stats(manifest: dict) -> None:
    print(f"template splits: {manifest['template_splits'] or '(by instance: fewer than 3 templates)'}")
    if manifest.get("preliminary"):
        print("note: fewer than 20 templates, so every number from this dataset is preliminary")
    print(f"{'split':6s} {'rows':>5s} {'inj':>5s} {'benign':>6s} {'HN':>4s} {'seen':>5s} "
          f"{'unseen':>6s} templates")
    for split, st in manifest["stats"].items():
        n_hn = sum(st["hard_negatives"].values())
        print(f"{split:6s} {st['rows']:5d} {st['injected']:5d} {st['benign']:6d} {n_hn:4d} "
              f"{st['injected_seen_family']:5d} {st['injected_unseen_family']:6d} "
              f"{','.join(t for t in st['templates'] if t not in ('augment', 'probe'))}")
    print(f"source overlap between splits: {manifest['source_overlap_between_splits']}")


def cmd_stats(args) -> int:
    from .dataset import load_manifest

    manifest = load_manifest(data_dir(args.root))
    if args.json:
        print(json.dumps(manifest, indent=1))
    else:
        _print_stats(manifest)
        print(f"fingerprint {manifest['fingerprint']}")
    return 0


def cmd_train(args) -> int:
    from .detectors import DetectorUnavailable
    from .train import TrainConfig, train_modernbert

    cfg = TrainConfig(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, device=args.device, max_train_rows=args.max_train_rows,
                      **({"base_model": args.base_model} if args.base_model else {}))
    out = models_dir(args.root) / (args.out or f"modernbert-seed{args.seed}")
    try:
        train_modernbert(data_dir(args.root), out, cfg)
    except DetectorUnavailable as e:
        return _unavailable(e)
    return 0


def _load(args):
    from .detectors import load_detector

    return load_detector(args.detector, args.checkpoint, root=args.root, device=args.device,
                         batch_size=args.batch_size)


def cmd_calibrate(args) -> int:
    from .detectors import DetectorUnavailable
    from .evaluate import calibrate

    tag = args.tag or args.detector
    try:
        det = _load(args)
    except DetectorUnavailable as e:
        return _unavailable(e)
    cal = calibrate(tag, args.detector, args.checkpoint, det.score_segments, root=args.root,
                    target_fpr=args.target_fpr, device=det.device, force=args.force)
    print(f"{tag}: threshold {cal.threshold:.4f} (log-odds; p = {cal.threshold_prob:.6f}) at "
          f"target FPR {cal.target_fpr}")
    print(f"  val FPR {_pct(cal.val['fpr'])}, val recall {_pct(cal.val['recall'])}, "
          f"val AUROC {_fmt(cal.val['auroc'])}")
    return 0


def cmd_evaluate(args) -> int:
    from .detectors import DetectorUnavailable, load_calibrated
    from .evaluate import evaluate

    try:
        det, cal = load_calibrated(args.tag, args.root, args.device)
    except DetectorUnavailable as e:
        return _unavailable(e)
    rep = evaluate(cal, det.score_segments, root=args.root, split=args.split)
    m = rep[args.split]
    print(f"{args.tag} on {args.split} at the frozen threshold {cal.threshold:.4f}:")
    for key in ("fpr", "fpr_world_benign", "fpr_hard_negatives", "fpr_world_hard_negatives",
                "fpr_augmented_hard_negatives", "fpr_composed_negatives",
                "fpr_text_unseen_in_train", "recall", "recall_seen_families",
                "recall_unseen_families"):
        print(f"  {key:30s} {_pct(m[key])}")
    print(f"  {'auroc (seen / unseen)':30s} {_fmt(m['auroc'])} ({_fmt(m['auroc_seen_families'])} "
          f"/ {_fmt(m['auroc_unseen_families'])})")
    if not rep["dataset_matches_calibration"]:
        print("warning: the dataset changed since calibration", file=sys.stderr)
    return 0


def cmd_errors(args) -> int:
    from .evaluate import errors

    rep = errors(args.tag, root=args.root, split=args.split, limit=args.limit)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


def cmd_latency(args) -> int:
    from .dataset import load_calls
    from .detectors import DetectorUnavailable, load_calibrated
    from .latency import bench_forward, bench_tool_calls, hardware_info

    try:
        det, cal = load_calibrated(args.tag, args.root, args.device)
    except DetectorUnavailable as e:
        return _unavailable(e)
    calls = load_calls(data_dir(args.root))
    if args.max_calls:
        calls = calls[:args.max_calls]
    forward = bench_forward(det, args.lengths, args.batch_sizes, args.repeats)
    per_call = bench_tool_calls(det.score_segments, cal.threshold, calls)
    report = {"tag": args.tag, "detector": det.info(), "hardware": hardware_info(det.device),
              "forward": forward, "tool_calls": per_call, "n_tool_calls": len(calls)}
    path = reports_dir(args.root) / f"{args.tag}_latency_{det.device}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1) + "\n")
    print(f"{args.tag} on {det.device} ({report['hardware'].get('chip', report['hardware']['machine'])})")
    print(f"  {'tokens':>6s} {'batch':>5s} {'p50 ms':>8s} {'p95 ms':>8s}")
    for r in forward:
        print(f"  {r['input_tokens']:6d} {r['batch_size']:5d} {r['p50_ms']:8.2f} {r['p95_ms']:8.2f}")
    a = per_call["all"]
    print(f"  per read-tool call (n={a['n']}): p50 {a['p50_ms']:.2f} ms, p95 {a['p95_ms']:.2f} ms, "
          f"{per_call['segments_per_call']['mean']} segments per call")
    print(f"wrote {path}")
    return 0


def cmd_e2e_proxy(args) -> int:
    from .detectors import DetectorUnavailable
    from .evaluate import e2e_proxy

    tags, rc = [], 0
    for tag in args.tags:
        try:
            from .arms import resolve_scorer

            resolve_scorer(tag, args.root, args.device)
            tags.append(tag)
        except DetectorUnavailable as e:
            rc = _unavailable(e)
    if not tags:
        return rc or EXIT_UNAVAILABLE
    rep = e2e_proxy(tags, world_seed=args.world_seed, instances=args.instances,
                    behavior=args.behavior)
    path = reports_dir(args.root) / f"e2e_proxy_{'_'.join(tags)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    for tag, m in rep["arms"].items():
        print(f"{tag}: {m['episodes']} episodes with ref-{args.behavior} on world seed "
              f"{args.world_seed}")
        print(f"  benign episodes with a redaction   {_pct(m['benign_episodes_with_redaction'])}")
        print(f"  payload reached model (exposed)    {_pct(m['payload_reached_model_given_exposed'])}")
        print(f"    seen families                    {_pct(m['payload_reached_model_seen_families'])}")
        print(f"    unseen families                  {_pct(m['payload_reached_model_unseen_families'])}")
        lat = m["filter_latency_per_call"]
        ep = m["filter_ms_per_episode"]
        print(f"  filter per call p50/p95 {lat['p50_ms']}/{lat['p95_ms']} ms; per episode p50/p95 "
              f"{ep['p50_ms']}/{ep['p95_ms']} ms; {m['read_calls_per_episode']:.2f} read calls "
              f"per episode")
        if m["benign_texts_redacted"]:
            print("  benign texts redacted:")
            for text, n in list(m["benign_texts_redacted"].items())[:10]:
                print(f"    {n:3d}x {text[:100]!r}")
    print(f"wrote {path}")
    return rc if not tags else 0


def cmd_check(args) -> int:
    from .arms import DETECTOR_ARMS
    from .detectors import DetectorUnavailable, calibration_path, load_calibrated

    ok = True
    for name, arm in DETECTOR_ARMS.items():
        path = calibration_path(arm.detector, args.root)
        status = "calibrated" if path.exists() else "NOT calibrated"
        if path.exists() and args.load:
            try:
                det, cal = load_calibrated(arm.detector, args.root, args.device)
                status += f", weights load on {det.device}"
            except DetectorUnavailable as e:
                status += f", weights NOT loadable: {e}"
                ok = False
        ok &= path.exists()
        print(f"{name:28s} {status} ({path})")
    return 0 if ok else EXIT_UNAVAILABLE


def cmd_all(args) -> int:
    """Everything that can run here, in order, skipping what is unavailable."""
    from .detectors import calibration_path

    root = args.root
    if args.rebuild or not (data_dir(root) / "manifest.json").exists():
        cmd_build_data(_ns(args, world_seeds=[1, 2, 3], instances=6, instance_seed=0,
                           split_seed=0, templates=None, val_templates=None,
                           test_templates=None, no_probes=False, no_augment=False,
                           no_compose=False))
    if not (models_dir(root) / "modernbert-seed0").is_dir():
        rc = cmd_train(_ns(args, seed=0, epochs=3, batch_size=16, lr=5e-5, max_train_rows=None,
                           base_model=None, out=None))
        if rc:
            return rc
    rc = 0
    for detector in ("modernbert", "promptguard2"):
        common = dict(detector=detector, checkpoint=None, tag=detector, batch_size=32)
        if not calibration_path(detector, root).exists():
            r = cmd_calibrate(_ns(args, target_fpr=0.01, force=False, **common))
            if r:
                rc = r
                continue
        cmd_evaluate(_ns(args, split="test", **common))
        cmd_latency(_ns(args, lengths=[32, 128, 256, 512], batch_sizes=[1, 8, 32], repeats=20,
                        max_calls=None, **common))
    cmd_e2e_proxy(_ns(args, tags=["modernbert", "promptguard2"], world_seed=0, instances=2,
                      behavior="gold"))
    return rc


def _ns(args, **kw) -> argparse.Namespace:
    return argparse.Namespace(**{**vars(args), **kw})


def main(argv: list[str] | None = None) -> int:
    from ..cli import load_dotenv

    p = argparse.ArgumentParser(prog="injbench-detect", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None,
                   help=f"artifact directory (default: $INJBENCH_DETECT_DIR or {detect_root()})")
    p.add_argument("--device", default=None, help="auto (default), mps, cuda or cpu")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-data", help="build the labeled segment dataset (no torch needed)")
    b.add_argument("--world-seeds", type=int, nargs="+", default=[1, 2, 3])
    b.add_argument("--instances", type=int, default=6, help="instances per template per world")
    b.add_argument("--instance-seed", type=int, default=0)
    b.add_argument("--split-seed", type=int, default=0)
    b.add_argument("--templates", nargs="+", help="default: every template")
    b.add_argument("--val-templates", nargs="+")
    b.add_argument("--test-templates", nargs="+")
    b.add_argument("--no-probes", action="store_true")
    b.add_argument("--no-augment", action="store_true")
    b.add_argument("--no-compose", action="store_true", help="no format-matched negatives")
    b.set_defaults(fn=cmd_build_data)

    s = sub.add_parser("stats", help="print the dataset card numbers")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_stats)

    t = sub.add_parser("train", help="fine-tune ModernBERT-base")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--max-train-rows", type=int, default=None)
    t.add_argument("--base-model", default=None, help="default answerdotai/ModernBERT-base")
    t.add_argument("--out", default=None, help="name under models/ (default modernbert-seed<N>)")
    t.set_defaults(fn=cmd_train)

    def detector_args(sp):
        sp.add_argument("--detector", default="modernbert",
                        help="modernbert, promptguard2 or promptguard2-22m")
        sp.add_argument("--checkpoint", default=None,
                        help="modernbert checkpoint name under models/ (default modernbert-seed0)")
        sp.add_argument("--batch-size", type=int, default=32)

    c = sub.add_parser("calibrate", help="choose and freeze the threshold on validation data")
    detector_args(c)
    c.add_argument("--tag", default=None, help="calibration name (default: the detector name)")
    c.add_argument("--target-fpr", type=float, default=0.01)
    c.add_argument("--force", action="store_true", help="overwrite a frozen threshold")
    c.set_defaults(fn=cmd_calibrate)

    e = sub.add_parser("evaluate", help="test metrics at the frozen threshold")
    e.add_argument("--tag", default="modernbert")
    e.add_argument("--split", default="test", choices=["val", "test"])
    e.set_defaults(fn=cmd_evaluate)

    er = sub.add_parser("errors", help="false positives and negatives (prints texts)")
    er.add_argument("--tag", default="modernbert")
    er.add_argument("--split", default="test", choices=["val", "test"])
    er.add_argument("--limit", type=int, default=10)
    er.set_defaults(fn=cmd_errors)

    lt = sub.add_parser("latency", help="p50/p95 latency per forward pass and per tool call")
    lt.add_argument("--tag", default="modernbert")
    lt.add_argument("--lengths", type=int, nargs="+", default=[32, 128, 256, 512])
    lt.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    lt.add_argument("--repeats", type=int, default=20)
    lt.add_argument("--max-calls", type=int, default=None)
    lt.set_defaults(fn=cmd_latency)

    ep = sub.add_parser("e2e-proxy", help="arms inside the benchmark with scripted agents")
    ep.add_argument("--tags", nargs="+", default=["modernbert", "promptguard2"])
    ep.add_argument("--world-seed", type=int, default=0, help="0 is the benchmark world")
    ep.add_argument("--instances", type=int, default=2)
    ep.add_argument("--behavior", default="gold", choices=["gold", "gullible"])
    ep.set_defaults(fn=cmd_e2e_proxy)

    ck = sub.add_parser("check", help="are the arms calibrated (and loadable with --load)?")
    ck.add_argument("--load", action="store_true")
    ck.set_defaults(fn=cmd_check)

    al = sub.add_parser("all", help="build, train, calibrate, evaluate, time; skip the unavailable")
    al.add_argument("--rebuild", action="store_true")
    al.set_defaults(fn=cmd_all)

    args = p.parse_args(argv)
    load_dotenv()
    ensure_hf_env()
    # Arms inside an episode only see the environment, so export the choices made here.
    if args.root:
        os.environ[ROOT_ENV] = str(args.root)
    if args.device:
        os.environ[DEVICE_ENV] = args.device
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
