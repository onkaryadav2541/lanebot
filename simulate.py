#!/usr/bin/env python3
"""
simulate.py -- Closed-loop software-in-the-loop (SIL) evaluation.

A recorded video is open loop: the frames do not react to the steering
commands, so it can only measure perception quality. Hypotheses H1 and H2 are
statements about *closed-loop* behaviour:

    H1  histogram detection gives smoother steering on curves, with a lower
        oscillation amplitude than Canny-Hough
    H2  latency above 100 ms significantly degrades PID performance, below
        80 ms improves responsiveness

To test these without risking the vehicle, this module closes the loop around
a differential-drive kinematic model of the car. Crucially it does not fake
the perception: each simulation step renders a synthetic camera image from
the vehicle state and pushes it through the real Preprocessor, the real
detector and the real controller. Only the plant is simulated.

    vehicle state -> rendered frame -> vision.py -> detectors.py
         ^                                              |
         |                                        controller.py
         +------------- kinematic model <---- delay buffer (latency)

Usage:
    python3 simulate.py                        # single run, prints metrics
    python3 simulate.py --compare-controllers  # PID vs bang-bang   (H1)
    python3 simulate.py --compare-algorithms   # histogram vs Hough (H1)
    python3 simulate.py --latency-sweep        # 0..160 ms          (H2)
"""

import argparse
import json
import os
import time
from collections import deque

import numpy as np

import camera as cam
from config import Config
from controller import SpeedMixer, create_controller
from detectors import create_detector
from vision import Preprocessor

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


# --------------------------------------------------------------------------
# Vehicle and track model
# --------------------------------------------------------------------------
V_MAX = 0.60           # m/s at 100 % duty (measured on the 6 V pack)
TRACK_WIDTH = 0.145    # m, distance between the left and right wheel pairs
PX_PER_M = 320.0       # image scale: 1 m of lateral offset ~ full frame width
LOOKAHEAD = 0.18       # m ahead where the ROI band actually looks
LANE_HALF_WIDTH = 0.20 # m, half of the physical lane


class Track:
    """Sequence of straight and curved segments, parametrised by arc length."""

    def __init__(self, layout="mixed"):
        if layout == "straight":
            self.segments = [(4.0, 0.0)]
        elif layout == "curve":
            self.segments = [(1.0, 0.0), (6.0, 1.4)]
        else:  # mixed: straight, right curve, straight, left curve, ...
            self.segments = [(1.5, 0.0), (2.5, 1.6), (1.5, 0.0),
                             (2.5, -1.6), (1.5, 0.0), (2.0, 2.2), (2.0, 0.0)]
        self.length = sum(s[0] for s in self.segments)

    def curvature(self, s):
        s = s % self.length
        acc = 0.0
        for length, kappa in self.segments:
            if s < acc + length:
                return kappa
            acc += length
        return self.segments[-1][1]


class Vehicle:
    """
    Differential-drive kinematics in lane-relative coordinates.

        y    lateral offset from the lane centre  (+ = right of centre) [m]
        psi  heading error relative to the lane tangent (+ = pointing right) [rad]
        s    distance travelled along the track [m]
    """

    def __init__(self, y0=0.06, psi0=0.0):
        self.y = y0
        self.psi = psi0
        self.s = 0.0

    def step(self, left_speed, right_speed, curvature, dt):
        v_l = left_speed * V_MAX
        v_r = right_speed * V_MAX
        v = 0.5 * (v_l + v_r)
        omega = (v_l - v_r) / TRACK_WIDTH        # + = turning right
        self.psi += (omega - v * curvature) * dt
        self.psi = float(np.clip(self.psi, -1.2, 1.2))
        self.y += v * np.sin(self.psi) * dt
        self.s += v * np.cos(self.psi) * dt
        return v

    def lane_shift_px(self):
        """Where the lane centre appears in the image, in pixels from centre."""
        lateral = self.y + LOOKAHEAD * np.tan(self.psi)
        return float(np.clip(-lateral * PX_PER_M, -150, 150))


# --------------------------------------------------------------------------
def run_simulation(cfg, algorithm=None, controller_name=None,
                   extra_latency_ms=0.0, duration=25.0, control_hz=15.0,
                   layout="mixed", noise=8, seed=0, verbose=False):
    """Returns a dict of time series and metrics for one closed-loop run."""
    np.random.seed(seed)
    cfg = Config.load(cfg) if isinstance(cfg, str) else cfg
    if algorithm:
        cfg.algorithm = algorithm
    if controller_name:
        cfg.control.controller = controller_name

    pre = Preprocessor(cfg.vision)
    detector = create_detector(cfg.algorithm, cfg.vision, pre)
    controller = create_controller(cfg.control)
    mixer = SpeedMixer(cfg.control)

    dt = 1.0 / control_hz
    # Every digital loop applies a command that was computed from an already
    # captured frame, so there is always at least one sample of dead time.
    # `extra_latency_ms` is added on top of that inherent delay.
    delay_steps = 1 + int(round((extra_latency_ms / 1000.0) / dt))
    command_buffer = deque([(cfg.control.base_speed, cfg.control.base_speed)]
                           * max(0, delay_steps), maxlen=max(1, delay_steps + 1))

    track = Track(layout)
    car = Vehicle()
    target = cfg.camera.width / 2.0
    half_width = cfg.camera.width / 2.0

    t_series, y_series, e_series, steer_series, detected = [], [], [], [], []
    lost = 0
    steps = int(duration * control_hz)

    for k in range(steps):
        kappa = track.curvature(car.s)
        frame = cam.synthetic_frame(
            cfg.camera.width, cfg.camera.height,
            offset=car.lane_shift_px(),
            curve=float(np.clip(kappa * 0.45, -1.0, 1.0)),
            noise=noise)

        p = pre.process(frame)
        result = detector.detect(p, cfg.camera.width)

        if result.detected:
            error_px = result.lane_center - target
            steer = controller.update(error_px, half_width, dt)
            lost = 0
        else:
            error_px = float("nan")
            lost += 1
            steer = 0.0 if lost > cfg.control.lost_lane_hold_frames else steer_series[-1] if steer_series else 0.0

        left, right = mixer.mix(steer)

        if delay_steps > 0:
            command_buffer.append((left, right))
            applied_left, applied_right = command_buffer[0]
        else:
            applied_left, applied_right = left, right

        car.step(applied_left, applied_right, kappa, dt)

        t_series.append(k * dt)
        y_series.append(car.y)
        e_series.append(error_px)
        steer_series.append(steer)
        detected.append(int(result.detected))

        if abs(car.y) > LANE_HALF_WIDTH * 2.5:
            if verbose:
                print("    lane departure at t = %.1f s" % (k * dt))
            break

    y = np.asarray(y_series)
    inside = np.abs(y) <= LANE_HALF_WIDTH
    sign = np.sign(y)
    sign[sign == 0] = 1
    crossings = int(np.sum(sign[1:] != sign[:-1])) if y.size > 1 else 0
    completed = len(y_series) == steps

    metrics = {
        "algorithm": detector.name,
        "controller": controller.name,
        "extra_latency_ms": extra_latency_ms,
        "completed": bool(completed),
        "distance_m": round(float(car.s), 2),
        "time_s": round(len(y_series) * dt, 2),
        "mean_abs_offset_cm": round(float(np.mean(np.abs(y)) * 100), 2),
        "rms_offset_cm": round(float(np.sqrt(np.mean(y ** 2)) * 100), 2),
        "max_abs_offset_cm": round(float(np.max(np.abs(y)) * 100), 2),
        "oscillation_amplitude_cm": round(float(np.std(y) * 100), 2),
        "centre_crossings_per_m": round(crossings / max(0.01, car.s), 2),
        "in_lane_fraction": round(float(np.mean(inside)), 4),
        "detection_rate": round(float(np.mean(detected)), 4),
        "steer_activity": round(float(np.mean(np.abs(np.diff(steer_series)))), 4)
        if len(steer_series) > 1 else 0.0,
    }
    return {"t": t_series, "y": y_series, "e": e_series,
            "steer": steer_series, "metrics": metrics}


# --------------------------------------------------------------------------
def _plot_runs(runs, labels, path, title):
    if not HAVE_PLT:
        return
    plt.figure(figsize=(10, 4.5))
    for run, label in zip(runs, labels):
        plt.plot(run["t"], np.asarray(run["y"]) * 100, linewidth=1.2, label=label)
    plt.axhline(0, color="k", linewidth=0.6)
    plt.axhline(LANE_HALF_WIDTH * 100, color="r", linestyle="--", linewidth=0.7)
    plt.axhline(-LANE_HALF_WIDTH * 100, color="r", linestyle="--", linewidth=0.7)
    plt.xlabel("time (s)")
    plt.ylabel("lateral offset from lane centre (cm)")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("  plot -> %s" % path)


def _print_table(rows, keys, title):
    print("\n" + title)
    print("-" * 96)
    print("  " + "".join("%-22s" % k for k in keys))
    print("-" * 96)
    for r in rows:
        print("  " + "".join("%-22s" % r.get(k, "") for k in keys))
    print("-" * 96)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Closed-loop SIL evaluation")
    ap.add_argument("--config")
    ap.add_argument("--algo", choices=["histogram", "canny_hough"])
    ap.add_argument("--controller", choices=["pid", "bangbang"])
    ap.add_argument("--latency", type=float, default=0.0,
                    help="extra actuation latency in ms")
    ap.add_argument("--duration", type=float, default=25.0)
    ap.add_argument("--hz", type=float, default=15.0)
    ap.add_argument("--layout", default="mixed",
                    choices=["straight", "curve", "mixed"])
    ap.add_argument("--compare-controllers", action="store_true")
    ap.add_argument("--compare-algorithms", action="store_true")
    ap.add_argument("--latency-sweep", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, "sim_%s" % stamp)
    os.makedirs(out_dir, exist_ok=True)
    base = Config.load(args.config) if args.config else Config()
    results = {}

    def cfg():
        return Config.load(args.config) if args.config else Config()

    if args.compare_controllers:
        print("\n== H1: PID vs bang-bang (%s, %s track) ==" %
              (args.algo or base.algorithm, args.layout))
        runs, labels, rows = [], [], []
        for name in ("bangbang", "pid"):
            r = run_simulation(cfg(), algorithm=args.algo, controller_name=name,
                               duration=args.duration, control_hz=args.hz,
                               layout=args.layout)
            runs.append(r)
            labels.append(name)
            rows.append(r["metrics"])
        _print_table(rows, ["controller", "mean_abs_offset_cm",
                            "oscillation_amplitude_cm", "max_abs_offset_cm",
                            "in_lane_fraction", "steer_activity"],
                     "Controller comparison")
        _plot_runs(runs, labels, os.path.join(out_dir, "controllers.png"),
                   "Lateral offset: bang-bang vs PID")
        results["controllers"] = rows

    if args.compare_algorithms:
        print("\n== H1: histogram vs Canny-Hough (%s controller) ==" %
              (args.controller or base.control.controller))
        runs, labels, rows = [], [], []
        for algo in ("canny_hough", "histogram"):
            r = run_simulation(cfg(), algorithm=algo,
                               controller_name=args.controller,
                               duration=args.duration, control_hz=args.hz,
                               layout=args.layout)
            runs.append(r)
            labels.append(algo)
            rows.append(r["metrics"])
        _print_table(rows, ["algorithm", "detection_rate", "mean_abs_offset_cm",
                            "oscillation_amplitude_cm", "in_lane_fraction",
                            "centre_crossings_per_m"],
                     "Algorithm comparison (closed loop)")
        _plot_runs(runs, labels, os.path.join(out_dir, "algorithms.png"),
                   "Lateral offset: Algorithm A vs Algorithm B")
        results["algorithms"] = rows

    if args.latency_sweep:
        print("\n== H2: effect of end-to-end latency on PID performance ==")
        print("   note: the delay buffer works in whole control periods, so at"
              " %.0f Hz the\n   resolution is %.0f ms. Run with --hz 30 for a"
              " finer sweep."
              % (args.hz, 1000.0 / args.hz))
        runs, labels, rows = [], [], []
        for lat in (0, 40, 60, 80, 100, 120, 160):
            r = run_simulation(cfg(), algorithm=args.algo,
                               controller_name="pid", extra_latency_ms=lat,
                               duration=args.duration, control_hz=args.hz,
                               layout=args.layout)
            runs.append(r)
            labels.append("%d ms" % lat)
            rows.append(r["metrics"])
            print("  %4d ms  osc %5.2f cm  mean|y| %5.2f cm  in-lane %.2f  %s"
                  % (lat, r["metrics"]["oscillation_amplitude_cm"],
                     r["metrics"]["mean_abs_offset_cm"],
                     r["metrics"]["in_lane_fraction"],
                     "completed" if r["metrics"]["completed"] else "DEPARTED"))
        _plot_runs(runs, labels, os.path.join(out_dir, "latency_sweep.png"),
                   "Effect of added latency on closed-loop PID tracking")
        if HAVE_PLT:
            plt.figure(figsize=(7, 4))
            plt.plot([r["extra_latency_ms"] for r in rows],
                     [r["oscillation_amplitude_cm"] for r in rows], "o-")
            plt.axvline(80, color="g", linestyle="--", label="H2 target 80 ms")
            plt.axvline(100, color="r", linestyle="--", label="H2 limit 100 ms")
            plt.xlabel("added latency (ms)")
            plt.ylabel("oscillation amplitude (cm)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "latency_vs_oscillation.png"), dpi=150)
            plt.close()
        results["latency"] = rows

    if not (args.compare_controllers or args.compare_algorithms or args.latency_sweep):
        r = run_simulation(cfg(), algorithm=args.algo,
                           controller_name=args.controller,
                           extra_latency_ms=args.latency,
                           duration=args.duration, control_hz=args.hz,
                           layout=args.layout, verbose=True)
        print(json.dumps(r["metrics"], indent=2))
        _plot_runs([r], ["%s / %s" % (r["metrics"]["algorithm"],
                                      r["metrics"]["controller"])],
                   os.path.join(out_dir, "run.png"), "Closed-loop run")
        results["single"] = r["metrics"]

    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nResults in %s\n" % out_dir)


if __name__ == "__main__":
    main()
