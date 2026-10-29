"""Run every spacing case concurrently, with isolated snapshots and plots."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys


CAMPAIGN = Path(__file__).resolve().parent
REPO = CAMPAIGN.parents[1]


def run_case(case, check_only=False):
    # Each child process owns its cwd and main.RESULTS_DIR. Several plotting
    # helpers use cwd/results, while main.py uses its module-level RESULTS_DIR.
    work = CAMPAIGN / "runs" / case.name
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(REPO))
    import main

    main.RESULTS_DIR = str(work / "results")
    sys.argv = [
        str(REPO / "main.py"),
        "--config", str(case / "config.txt"),
        "--experiment-config", str(case / "experimental_config.txt"),
        "--post-config", str(CAMPAIGN / "post_config.txt"),
        "--outdir", str(work / "output"),
    ]
    args = main.parse_args()
    if check_only:
        assert (args.nx, args.ny) == (256, 512)
        print(f"{case.name}: {args.nx}x{args.ny}, "
              f"L/D={args.cylinder_downstream_spacing_ratio:g}")
        return
    if any((work / "output").glob("snap_*.npz")):
        raise RuntimeError(f"Snapshots already exist in {work / 'output'}. "
                           "Move that run folder before starting a fresh run.")
    grid, loaded = main.get_runtime_grid(args)
    main.run(args, grid=grid, grid_loaded_from_file=loaded)


def launch(case, check_only):
    command = [sys.executable, str(Path(__file__).resolve()), "--case", case.name]
    if check_only:
        command.append("--check")
    work = CAMPAIGN / "runs" / case.name
    work.mkdir(parents=True, exist_ok=True)
    log = work / ("check.log" if check_only else "run.log")
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    return case.name, result.returncode, log


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4,
                        help="Concurrent cases (default: all four; use 1 for sequential)")
    parser.add_argument("--check", action="store_true",
                        help="Parse and validate configs without running simulations")
    parser.add_argument("--case", help="Run just this case directory")
    options = parser.parse_args()
    cases = sorted(p.parent for p in CAMPAIGN.glob("*/experimental_config.txt"))
    if options.jobs < 1:
        parser.error("--jobs must be positive")
    if options.case:
        matching = [p for p in cases if p.name == options.case]
        if not matching:
            parser.error(f"Unknown case: {options.case}")
        run_case(matching[0], options.check)
        return 0
    if not cases:
        parser.error("No case configs found")
    failed = False
    with ThreadPoolExecutor(max_workers=options.jobs) as pool:
        futures = [pool.submit(launch, case, options.check) for case in cases]
        for future in as_completed(futures):
            name, status, log = future.result()
            failed |= status != 0
            print(f"{name}: {'OK' if status == 0 else 'FAILED'} | {log}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(cli())
