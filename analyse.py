#!/usr/bin/env python3
"""
analyse.py -- Runs every offline analysis in one command.

The driving runs have to be done by hand, because the motors are armed with a
key press. Everything after that is automated here: figures, comparison
tables, the algorithm and threshold sweep, the closed-loop simulation and the
Ziegler-Nichols search. All outputs are collected into a single folder so the
thesis has one place to draw from.

Usage
-----
After recording the runs (see below), simply:

    python3 analyse.py

It finds the newest run of each configuration automatically. To pick specific
runs instead:

    python3 analyse.py --pid runs/pid_r1_* --bangbang runs/bb_r1_* \\
                       --hough runs/hough_r1_*

Skip the slow parts if you are in a hurry:

    python3 analyse.py --quick        # no simulation, no Ziegler-Nichols

Recording the runs first
------------------------
    python3 main.py --controller pid      --no-graphs --every 4 --record --name pid_r1
    python3 main.py --controller bangbang --no-graphs --every 4 --name bb_r1
    python3 main.py --algo canny_hough    --no-graphs --every 4 --name hough_r1

Repeat each with --name pid_r2, pid_r3 and so on for mean +/- SD statistics.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def newest(pattern):
    """Most recently modified run folder matching the pattern."""
    hits = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def all_matching(pattern):
    return sorted(p for p in glob.glob(pattern) if os.path.isdir(p))


def has_frames(path):
    return path and os.path.exists(os.path.join(path, "frames.csv"))


def run(step, args, out_dir, log):
    """Runs one analysis step, recording success or failure rather than stopping."""
    print("\n" + "=" * 62)
    print("  %s" % step)
    print("=" * 62)
    cmd = [sys.executable] + args
    try:
        proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                              timeout=3600)
        sys.stdout.write(proc.stdout[-4000:])
        if proc.returncode != 0:
            sys.stdout.write(proc.stderr[-2000:])
            log.append((step, "FAILED"))
            print("  -> %s failed, continuing with the rest" % step)
            return False
        log.append((step, "ok"))
        return True
    except subprocess.TimeoutExpired:
        log.append((step, "TIMED OUT"))
        print("  -> %s timed out" % step)
        return False


def collect(out_dir):
    """Copies every figure and table produced into one flat folder."""
    figures = os.path.join(out_dir, "figures")
    tables = os.path.join(out_dir, "tables")
    os.makedirs(figures, exist_ok=True)
    os.makedirs(tables, exist_ok=True)

    n_fig = n_tab = 0
    for root, _dirs, files in os.walk(os.path.join(HERE, "results")):
        if os.path.abspath(root).startswith(os.path.abspath(out_dir)):
            continue
        for f in files:
            src = os.path.join(root, f)
            tag = os.path.basename(root)
            if f.endswith((".png", ".pdf")):
                dst = os.path.join(figures, "%s__%s" % (tag, f))
                shutil.copy2(src, dst)
                n_fig += 1
            elif f.endswith((".md", ".tex", ".csv", ".json")):
                dst = os.path.join(tables, "%s__%s" % (tag, f))
                shutil.copy2(src, dst)
                n_tab += 1

    # figures written inside the run folders themselves
    for root, _dirs, files in os.walk(os.path.join(HERE, "runs")):
        if os.path.basename(root) != "figures":
            continue
        tag = os.path.basename(os.path.dirname(root))
        for f in files:
            if f.endswith((".png", ".pdf")):
                shutil.copy2(os.path.join(root, f),
                             os.path.join(figures, "%s__%s" % (tag, f)))
                n_fig += 1
    return n_fig, n_tab


def main():
    ap = argparse.ArgumentParser(description="Run every offline analysis")
    ap.add_argument("--pid", default="runs/pid_r*")
    ap.add_argument("--bangbang", default="runs/bb_r*")
    ap.add_argument("--hough", default="runs/hough_r*")
    ap.add_argument("--quick", action="store_true",
                    help="skip the simulation and Ziegler-Nichols steps")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    pid = newest(args.pid)
    bang = newest(args.bangbang)
    hough = newest(args.hough)

    print("\nRuns found")
    print("  PID         : %s" % (pid or "NONE"))
    print("  Bang-bang   : %s" % (bang or "NONE"))
    print("  Canny-Hough : %s" % (hough or "NONE"))

    if not has_frames(pid):
        raise SystemExit(
            "\nNo PID run found. Record one first:\n"
            "  python3 main.py --controller pid --no-graphs --every 4 "
            "--record --name pid_r1\n")

    # warn early if a run was mostly stationary
    for name, path in (("PID", pid), ("Bang-bang", bang), ("Hough", hough)):
        summary = os.path.join(path, "summary.json") if path else None
        if summary and os.path.exists(summary):
            with open(summary) as fh:
                data = json.load(fh)
            if data.get("WARNING"):
                print("\n  ! %s run: %s" % (name, data["WARNING"]))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(HERE, args.out, "analysis_%s" % stamp)
    os.makedirs(out_dir, exist_ok=True)
    log = []

    # 1 - detailed figures for the PID run on its own
    run("Figures for the PID run", ["report.py", "--run", pid], out_dir, log)

    # 2 - PID against bang-bang
    if has_frames(bang):
        run("PID versus bang-bang",
            ["report.py", "--run", pid, "--run", bang,
             "--labels", "PID", "Bang-bang", "--name", "pid_vs_bangbang"],
            out_dir, log)

    # 3 - histogram against Canny-Hough
    if has_frames(hough):
        run("Histogram versus Canny-Hough",
            ["report.py", "--run", pid, "--run", hough,
             "--labels", "Histogram", "Canny-Hough", "--name", "algo_comparison"],
            out_dir, log)

    # 4 - repeatability, when several runs of a configuration exist
    groups = []
    for label, pattern in (("PID", args.pid), ("Bang-bang", args.bangbang),
                           ("Canny-Hough", args.hough)):
        found = [p for p in all_matching(pattern) if has_frames(p)]
        if found:
            groups.append(["--group", label] + found)
    if groups and sum(len(g) - 2 for g in groups) > len(groups):
        flat = []
        for g in groups:
            flat += g
        run("Repeatability across runs", ["aggregate.py"] + flat, out_dir, log)
    else:
        print("\n  (only one run per configuration - skipping repeatability; "
              "record pid_r2, pid_r3 etc. for mean +/- SD)")

    # 5 - offline sweep over thresholds and histogram sources, on the video
    video = os.path.join(pid, "raw.avi")
    if os.path.exists(video):
        run("Threshold and algorithm sweep (H3)",
            ["benchmark.py", "--video", video, "--thresholds", "--sources"],
            out_dir, log)
    else:
        print("\n  (no raw.avi in the PID run - record with --record to enable "
              "the threshold sweep)")

    if not args.quick:
        # 6 - closed-loop simulation for H1 and H2
        run("Closed-loop simulation (H1, H2)",
            ["simulate.py", "--compare-controllers", "--compare-algorithms",
             "--latency-sweep", "--hz", "30"], out_dir, log)

        # 7 - Ziegler-Nichols tuning evidence
        run("Ziegler-Nichols gain search",
            ["tune_pid.py", "--mode", "zn"], out_dir, log)

    # 8 - analysis of the recorded error signal
    frames = os.path.join(pid, "frames.csv")
    run("Recorded-run analysis",
        ["tune_pid.py", "--mode", "analyse", "--csv", frames], out_dir, log)

    n_fig, n_tab = collect(out_dir)

    print("\n" + "=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    for step, status in log:
        print("  %-8s %s" % (status, step))
    print("-" * 62)
    print("  %d figures and %d tables collected" % (n_fig, n_tab))
    print("  figures : %s" % os.path.join(out_dir, "figures"))
    print("  tables  : %s" % os.path.join(out_dir, "tables"))
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
