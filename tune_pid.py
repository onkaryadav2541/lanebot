#!/usr/bin/env python3
"""
tune_pid.py -- PID gain determination (exposé section 3, "Ziegler-Nichols
method and iterative field testing").

Two modes:

  1. --mode zn      Automatic ultimate-gain search in the closed-loop
                    simulator. Ki and Kd are set to zero, Kp is increased in
                    steps until the lateral offset shows sustained
                    oscillation. From the critical gain Ku and the
                    oscillation period Tu the classic Ziegler-Nichols table
                    gives the starting gains:

                        P     : Kp = 0.50 Ku
                        PI    : Kp = 0.45 Ku, Ti = Tu/1.2
                        PID   : Kp = 0.60 Ku, Ti = Tu/2, Td = Tu/8
                        no-ov : Kp = 0.20 Ku, Ti = Tu/2, Td = Tu/3

                    with Ki = Kp/Ti and Kd = Kp*Td. These are a starting
                    point, not the final answer -- ZN is deliberately
                    aggressive, and the thesis should report both the ZN
                    values and the values after field refinement.

  2. --mode analyse Reads frames.csv from a real run and reports overshoot,
                    settling behaviour, dominant oscillation period and
                    saturation time, then suggests which gain to change.

Usage:
    python3 tune_pid.py --mode zn
    python3 tune_pid.py --mode zn --algo canny_hough --layout curve
    python3 tune_pid.py --mode analyse --csv runs/histogram_pid_2026.../frames.csv
"""

import argparse
import csv
import json
import os

import numpy as np

from config import Config
from simulate import run_simulation


# --------------------------------------------------------------------------
def dominant_period(signal, dt):
    """Estimate the oscillation period from mean zero-crossing spacing."""
    y = np.asarray(signal, dtype=float)
    y = y - np.mean(y)
    sign = np.sign(y)
    sign[sign == 0] = 1
    idx = np.where(sign[1:] != sign[:-1])[0]
    if len(idx) < 3:
        return None
    spacing = np.diff(idx) * dt
    return float(2.0 * np.mean(spacing))          # half period -> full period


def sustained_oscillation(y, dt, min_amplitude=0.002):
    """
    A sustained oscillation is one whose amplitude neither decays nor grows.

    The two comparison windows are the second and the last third of the run,
    never the first third: the run starts from a deliberate lateral offset,
    and that initial transient would otherwise be mistaken for a large
    oscillation and make every gain look "damped" by comparison.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 60:
        return False, 0.0, None
    third = y.size // 3
    a1 = np.std(y[third:2 * third])
    a2 = np.std(y[2 * third:])
    period = dominant_period(y[2 * third:], dt)
    if a1 < 1e-6:
        return False, a2, period
    ratio = a2 / a1
    tail_amp, min_amplitude = a2, min_amplitude
    tail = np.asarray(y[2 * third:], dtype=float)
    tail = tail - np.mean(tail)
    sign = np.sign(tail)
    sign[sign == 0] = 1
    crossings = int(np.sum(sign[1:] != sign[:-1]))
    sustained = (0.65 < ratio < 1.8 and tail_amp > min_amplitude
                 and crossings >= 3 and period is not None)
    return sustained, a2, period


# --------------------------------------------------------------------------
def ziegler_nichols(args):
    cfg = Config.load(args.config) if args.config else Config()
    if args.algo:
        cfg.algorithm = args.algo
    cfg.control.controller = "pid"
    cfg.control.ki = 0.0
    cfg.control.kd = 0.0
    cfg.control.deadzone_px = 0.0                  # ZN needs a linear loop
    dt = 1.0 / args.hz

    print("\nZiegler-Nichols ultimate-gain search")
    print("  algorithm %s, track %s, %.0f Hz, %.0f s per trial"
          % (cfg.algorithm, args.layout, args.hz, args.duration))
    print("-" * 72)
    print("  %8s  %10s  %10s  %10s  %s" %
          ("Kp", "amp (cm)", "period (s)", "in-lane", "verdict"))
    print("-" * 72)

    ku = tu = None
    trials = []
    kp = args.kp_start
    while kp <= args.kp_max + 1e-9:
        cfg.control.kp = kp
        run = run_simulation(cfg, duration=args.duration, control_hz=args.hz,
                             layout=args.layout, extra_latency_ms=args.latency)
        y = run["y"]
        osc, amp, period = sustained_oscillation(y, dt)
        m = run["metrics"]
        verdict = "sustained" if osc else (
            "departed" if not m["completed"] else "damped")
        print("  %8.2f  %10.2f  %10s  %10.2f  %s"
              % (kp, amp * 100, "%.2f" % period if period else "-",
                 m["in_lane_fraction"], verdict))
        trials.append({"kp": kp, "amplitude_cm": round(amp * 100, 3),
                       "period_s": period, "verdict": verdict,
                       "in_lane": m["in_lane_fraction"]})
        if osc and period:
            ku, tu = kp, period
            break
        if not m["completed"] and kp > args.kp_start:
            ku, tu = kp, period or dominant_period(y, dt)
            print("  -> lane departure, treating this Kp as the stability limit")
            break
        kp += args.kp_step

    print("-" * 72)
    if ku is None or not tu:
        print("No sustained oscillation found up to Kp = %.2f." % args.kp_max)
        print("Raise --kp-max, or increase --duration so slow oscillations "
              "have time to develop.")
        return

    print("Critical gain      Ku = %.3f" % ku)
    print("Oscillation period Tu = %.3f s\n" % tu)

    rules = {
        "P":            (0.50 * ku, None, None),
        "PI":           (0.45 * ku, tu / 1.2, None),
        "classic PID":  (0.60 * ku, tu / 2.0, tu / 8.0),
        "no overshoot": (0.20 * ku, tu / 2.0, tu / 3.0),
    }
    print("  %-14s %8s %8s %8s" % ("rule", "Kp", "Ki", "Kd"))
    print("  " + "-" * 40)
    out = {"ku": ku, "tu": tu, "trials": trials, "rules": {}}
    for name, (kp_r, ti, td) in rules.items():
        ki = kp_r / ti if ti else 0.0
        kd = kp_r * td if td else 0.0
        print("  %-14s %8.3f %8.3f %8.3f" % (name, kp_r, ki, kd))
        out["rules"][name] = {"kp": round(kp_r, 4), "ki": round(ki, 4),
                              "kd": round(kd, 4)}

    kp_r = out["rules"]["classic PID"]
    print("\nVerification run with the classic PID gains:")
    cfg.control.kp = kp_r["kp"]
    cfg.control.ki = kp_r["ki"]
    cfg.control.kd = kp_r["kd"]
    cfg.control.deadzone_px = 12.0
    run = run_simulation(cfg, duration=args.duration, control_hz=args.hz,
                         layout="mixed", extra_latency_ms=args.latency)
    print(json.dumps(run["metrics"], indent=2))
    out["verification"] = run["metrics"]

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "ziegler_nichols.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nSaved to %s" % path)
    print("\nPut the chosen values into config.py (ControlConfig) or pass them "
          "on the command line:\n  python3 main.py --kp %.3f --ki %.3f --kd %.3f\n"
          % (kp_r["kp"], kp_r["ki"], kp_r["kd"]))


# --------------------------------------------------------------------------
def analyse(args):
    if not args.csv or not os.path.exists(args.csv):
        raise SystemExit("Provide --csv path/to/frames.csv")

    err, t, steer, sat = [], [], [], 0
    with open(args.csv) as fh:
        for row in csv.DictReader(fh):
            if row.get("detected") != "1":
                continue
            try:
                err.append(float(row["error_px"]))
                t.append(float(row["t_rel"]))
                s = float(row["steer"]) if row["steer"] != "" else 0.0
                steer.append(s)
                if abs(s) > 0.97:
                    sat += 1
            except (ValueError, KeyError):
                continue

    if len(err) < 20:
        raise SystemExit("Not enough valid rows in %s" % args.csv)

    err = np.asarray(err)
    t = np.asarray(t)
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 1 / 15.0
    period = dominant_period(err, dt)
    sign = np.sign(err)
    sign[sign == 0] = 1
    crossings = int(np.sum(sign[1:] != sign[:-1]))
    bias = float(np.mean(err))

    print("\nRecorded-run analysis: %s" % args.csv)
    print("-" * 60)
    print("  samples                 %d" % err.size)
    print("  mean error (bias)       %+.2f px" % bias)
    print("  mean |error|            %.2f px" % np.mean(np.abs(err)))
    print("  RMS error               %.2f px" % np.sqrt(np.mean(err ** 2)))
    print("  peak |error|            %.2f px" % np.max(np.abs(err)))
    print("  oscillation std         %.2f px" % np.std(err))
    print("  centre crossings        %d (%.2f per s)" %
          (crossings, crossings / max(0.01, t[-1] - t[0])))
    print("  dominant period         %s" % ("%.2f s" % period if period else "none"))
    print("  steer saturated         %.1f %% of frames" % (100.0 * sat / err.size))
    print("-" * 60)

    print("Suggestions:")
    if np.std(err) > 25 and period and period < 1.5:
        print("  * Fast oscillation around the centre -> reduce Kp by ~30 %,")
        print("    or increase Kd to damp it.")
    if abs(bias) > 12:
        print("  * Persistent offset of %+.1f px -> increase Ki, or check the" % bias)
        print("    camera alignment: a tilted camera looks exactly like this.")
    if sat / err.size > 0.25:
        print("  * The output saturates often -> Kp is too high for the speed,")
        print("    or base_speed is too high for the turning radius.")
    if np.std(err) < 8 and abs(bias) < 8 and (not period or period > 2.5):
        print("  * Tracking is stable. Raise base_speed and re-check; the gains")
        print("    are speed dependent.")
    print()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="PID tuning helper")
    ap.add_argument("--mode", choices=["zn", "analyse"], default="zn")
    ap.add_argument("--config")
    ap.add_argument("--algo", choices=["histogram", "canny_hough"])
    # A straight track with an initial offset is the correct experiment for
    # the ultimate-gain test: any oscillation that appears is caused by the
    # controller, not by the geometry of the track.
    ap.add_argument("--layout", default="straight",
                    choices=["straight", "curve", "mixed"])
    ap.add_argument("--latency", type=float, default=0.0,
                    help="extra latency in ms to include in the tuning trials")
    ap.add_argument("--hz", type=float, default=15.0)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--kp-start", type=float, default=0.2)
    ap.add_argument("--kp-step", type=float, default=0.25)
    ap.add_argument("--kp-max", type=float, default=12.0)
    ap.add_argument("--csv", help="frames.csv of a recorded run (analyse mode)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.mode == "zn":
        ziegler_nichols(args)
    else:
        analyse(args)


if __name__ == "__main__":
    main()
