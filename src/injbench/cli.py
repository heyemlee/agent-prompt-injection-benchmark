"""`injbench` command line."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from .attacks import FAMILIES, GOALS
from .defenses import DEFENSES
from .metrics import summarize, to_markdown
from .tasks import TEMPLATES


def _cmd_world(args) -> int:
    from .world import build_world

    world = build_world(args.seed)
    out = Path(args.out)
    if out.exists():
        out.unlink()
    dst = sqlite3.connect(out)
    world.conn.backup(dst)
    dst.close()
    print(f"wrote {out} (seed {args.seed}, fingerprint {world.fingerprint()})")
    return 0


def _cmd_list(args) -> int:
    print("templates:", ", ".join(TEMPLATES))
    print("attack families:", ", ".join(FAMILIES))
    print("attack goals:", ", ".join(GOALS))
    print("defenses:", ", ".join(DEFENSES))
    return 0


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines from a local .env file into the environment.

    Existing environment variables win. Values are never printed.
    """
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _cmd_run(args) -> int:
    from .runner import build_matrix, load_config, run_experiment
    from .world import build_world

    cfg = load_config(args.config)
    if args.model:
        unknown = set(args.model) - {m.name for m in cfg.models}
        if unknown:
            print(f"unknown model name(s): {sorted(unknown)}", file=sys.stderr)
            return 2
        cfg.models = [m for m in cfg.models if m.name in args.model]
    if args.concurrency:
        cfg.concurrency = args.concurrency
    if args.dry_run:
        specs = build_matrix(cfg, build_world(cfg.world_seed))
        by = Counter((s.model.name, s.defense.name, "benign" if s.attack is None else "attack")
                     for s in specs)
        print(f"{len(specs)} episodes")
        for (m, d, c), n in sorted(by.items()):
            print(f"  {m:20s} {d:12s} {c:7s} {n}")
        return 0
    run_dir = asyncio.run(run_experiment(cfg, args.out))
    print((run_dir / "summary.md").read_text())
    print(f"results in {run_dir}")
    return 0


def _cmd_report(args) -> int:
    run_dir = Path(args.run_dir)
    records = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text().splitlines()
               if line.strip()]
    summary = summarize(records)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    md = to_markdown(summary)
    (run_dir / "summary.md").write_text(md)
    print(md)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="injbench", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("world", help="dump a generated world to an SQLite file for inspection")
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--out", default="world.sqlite")
    w.set_defaults(fn=_cmd_world)
    ls = sub.add_parser("list", help="list templates, attacks and defenses")
    ls.set_defaults(fn=_cmd_list)
    r = sub.add_parser("run", help="run an experiment config")
    r.add_argument("config")
    r.add_argument("--out", default=None, help="run directory (default: runs/<name>-<time>)")
    r.add_argument("--dry-run", action="store_true", help="only print the episode matrix")
    r.add_argument("--model", action="append", help="run only this model (repeatable)")
    r.add_argument("--concurrency", type=int, help="override the config's concurrent episodes")
    r.set_defaults(fn=_cmd_run)
    rep = sub.add_parser("report", help="recompute the summary of a finished run")
    rep.add_argument("run_dir")
    rep.set_defaults(fn=_cmd_report)
    from .adaptive import register_cli as register_adaptive  # M4: repeated vs adaptive attacks

    register_adaptive(sub)
    args = parser.parse_args(argv)
    load_dotenv()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
