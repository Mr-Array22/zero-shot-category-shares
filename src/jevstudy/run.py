#!/usr/bin/env python3
"""One-command experiment: local baselines, optional Jev, metrics and charts."""
import argparse
import json
from pathlib import Path
import shutil

from . import baselines
from . import jev_runner
from . import report
from .study import digest

from .paths import FROZEN_DATA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=Path("runs/my-run"))
    parser.add_argument("--execute-jev",action="store_true",help="Opt in to billable Jev requests; key entered locally")
    parser.add_argument("--budget-usd",type=float,default=1.)
    parser.add_argument("--max-requests",type=int)
    parser.add_argument("--retry-failed",action="store_true")
    parser.add_argument("--pause",type=float,default=.05)
    args=parser.parse_args()
    if args.max_requests is not None and args.max_requests<1 or args.pause<0:
        parser.error("Use a positive request count and nonnegative pause")
    folder=args.run.resolve()
    if not (folder/"study.jsonl").exists():
        if folder.exists() and any(folder.iterdir()):
            parser.error("New run directory must be empty")
        folder.mkdir(parents=True,exist_ok=True)
        frozen=FROZEN_DATA
        for name in ("study.jsonl","requests.jsonl","data-manifest.json"):
            shutil.copy2(frozen/name,folder/name)
    baselines.run(folder)
    outcome=jev_runner.run(folder,budget=args.budget_usd,execute=args.execute_jev,
                           max_requests=args.max_requests,pause=args.pause,retry_failed=args.retry_failed)
    measured=report.build(folder)
    from .figures import draw
    draw(folder,measured)
    print(f"Metrics: {folder/'comparison.csv'}")
    if args.execute_jev and outcome["status"]!="complete":
        print("Jev run is incomplete. Repeat the same command with the same --run folder to resume.")
        raise SystemExit(2)


if __name__=="__main__":main()
