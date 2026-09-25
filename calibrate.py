#!/usr/bin/env python3
"""
calibrate.py -- "Is my car actually working?" tools.

Four modes:

  --mode place    Live view with the ROI band, the detected boundaries and the
                  measured lane width drawn on top, plus written advice on how
                  to move the camera. Use this while physically adjusting the
                  camera bracket, before anything else.

  --mode scale    Measures pixels per centimetre by clicking two points a known
                  distance apart (use the ruler on the calibration sheet, or
                  the known lane width). Writes calibration.json, which
                  report.py reads with --px-per-cm so the thesis can state the
                  lateral deviation in centimetres instead of pixels.

  --mode check    A 30-second automated health check: camera frame rate,
                  detection rate, end-to-end latency, CPU temperature and
                  throttling flags, GPIO availability. Prints a pass/fail
                  checklist. Run this before every test session and paste the
                  output into your lab notebook.

  --mode thermal  Runs the full pipeline for several minutes while logging CPU
                  temperature, clock frequency, throttling flags and frame
                  rate. Produces the figure for the "CPU throttling and thermal
                  management" challenge named in the exposé.

Examples:
    python3 calibrate.py --mode place
    python3 calibrate.py --mode scale --known-mm 200
    python3 calibrate.py --mode check
    python3 calibrate.py --mode thermal --minutes 10
"""

import argparse
import json
import os
import subprocess
import time

import cv2
import numpy as np

import camera as cam
from config import Config
from detectors import create_detector
from vision import Preprocessor

FONT = cv2.FONT_HERSHEY_SIMPLEX
OK, WARN, BAD = (100, 220, 120), (60, 200, 250), (60, 60, 235)


# ----------------------------------------------------------------------
def pi_stat(command):
    try:
        out = subprocess.check_output(command, shell=True,
                                      stderr=subprocess.DEVNULL, timeout=3)
        return out.decode().strip()
    except Exception:
        return None


def cpu_temperature():
    raw = pi_stat("vcgencmd measure_temp")
    if raw and "=" in raw:
        try:
            return float(raw.split("=")[1].split("'")[0])
        except (IndexError, ValueError):
            pass
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as fh:
            return int(fh.read().strip()) / 1000.0
    except Exception:
        return None


def cpu_clock_mhz():
    raw = pi_stat("vcgencmd measure_clock arm")
    if raw and "=" in raw:
        try:
            return int(raw.split("=")[1]) / 1e6
        except (IndexError, ValueError):
            pass
    return None


def throttled_flags():
    """Decodes vcgencmd get_throttled into human-readable flags."""
    raw = pi_stat("vcgencmd get_throttled")
    if not raw or "=" not in raw:
        return None, []
    try:
        value = int(raw.split("=")[1], 16)
    except ValueError:
        return None, []
    bits = {
        0: "under-voltage now",
        1: "ARM frequency capped now",
        2: "throttled now",
        3: "soft temperature limit now",
        16: "under-voltage occurred",
        17: "ARM frequency capping occurred",
        18: "throttling occurred",
        19: "soft temperature limit occurred",
    }
    active = [text for bit, text in bits.items() if value & (1 << bit)]
    return value, active


# ======================================================================
def mode_place(cfg):
    """Live camera-placement assistant."""
    # The lane-width plausibility gate is disabled here. Its whole purpose is
    # to reject detections that do not match expected_lane_width -- but this
    # tool exists to MEASURE that width in the first place, so leaving the
    # gate on would reject the very lane the user is trying to calibrate
    # against, and report "no lane" for a perfectly good track.
    cfg.vision.lane_width_tolerance = None
    print("  (lane-width gate disabled for calibration)")
    src = cam.create_source(cfg.camera)
    pre = Preprocessor(cfg.vision)
    det = create_detector(cfg.algorithm, cfg.vision, pre)
    w = cfg.camera.width
    print("Camera placement. Keys: t = threshold mode, q = quit")

    widths, hits = [], []
    try:
        while True:
            ts, frame = src.read()
            if frame is None:
                break
            frame = cam.orient(frame, cfg.camera)
            p = pre.process(frame)
            r = det.detect(p, w)

            view = cv2.resize(frame, None, fx=2.0, fy=2.0,
                              interpolation=cv2.INTER_NEAREST)
            sc = 2.0
            top, bot = int(p.roi_top * sc), int(p.roi_bottom * sc)
            cv2.rectangle(view, (0, top), (view.shape[1] - 1, bot - 1),
                          (80, 200, 240), 1)
            cv2.putText(view, "ROI", (4, top - 4), FONT, 0.4, (80, 200, 240), 1)
            cv2.line(view, (int(w * sc / 2), top), (int(w * sc / 2), bot),
                     (160, 160, 160), 1)

            lane_px = None
            if r.detected:
                for x in (r.left_x, r.right_x):
                    if x is not None:
                        xi = int(x * sc)
                        cv2.line(view, (xi, top), (xi, bot), (235, 140, 60), 2)
                if r.left_x is not None and r.right_x is not None:
                    lane_px = r.right_x - r.left_x
                    widths.append(lane_px)
                cv2.circle(view, (int(r.lane_center * sc), (top + bot) // 2),
                           7, (120, 230, 140), -1)
            hits.append(1 if r.detected else 0)
            if len(hits) > 60:
                hits.pop(0)
            rate = float(np.mean(hits))

            # --- advice ---
            lines = []
            if not r.detected:
                lines.append(("NO LANE DETECTED", BAD))
                lines.append(("both lines must be inside the ROI band", WARN))
                lines.append(("tilt the camera down, or move the sheet closer", WARN))
            elif r.inferred:
                lines.append(("ONLY ONE BOUNDARY VISIBLE", WARN))
                lines.append(("move the camera back or raise it", WARN))
            elif lane_px is not None:
                frac = lane_px / float(w)
                if frac < 0.35:
                    lines.append(("LANE TOO NARROW IN IMAGE (%.0f%%)" % (100 * frac), WARN))
                    lines.append(("move the camera closer or lower it", WARN))
                elif frac > 0.85:
                    lines.append(("LANE FILLS THE IMAGE (%.0f%%)" % (100 * frac), WARN))
                    lines.append(("move the camera back or raise it", WARN))
                else:
                    lines.append(("GOOD PLACEMENT (lane = %.0f%% of width)"
                                  % (100 * frac), OK))
                lines.append(("lane width %.0f px, set expected_lane_width = %d"
                              % (lane_px, int(round(lane_px))), (220, 220, 220)))
            lines.append(("detection rate %.0f%% over the last %d frames"
                          % (100 * rate, len(hits)),
                          OK if rate > 0.95 else (WARN if rate > 0.7 else BAD)))
            lines.append(("threshold %s (T=%.0f), confidence %.2f  [t to change]"
                          % (cfg.vision.threshold_mode, p.threshold_used,
                             r.confidence), (170, 170, 170)))

            y = 18
            for text, colour in lines:
                cv2.putText(view, text, (8, y), FONT, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(view, text, (8, y), FONT, 0.42, colour, 1, cv2.LINE_AA)
                y += 18

            cv2.imshow("camera placement", view)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("t"):
                modes = ["fixed", "otsu", "adaptive"]
                i = modes.index(cfg.vision.threshold_mode)
                cfg.vision.threshold_mode = modes[(i + 1) % 3]
                pre = Preprocessor(cfg.vision)
                det = create_detector(cfg.algorithm, cfg.vision, pre)
    finally:
        src.release()
        cv2.destroyAllWindows()

    if widths:
        med = float(np.median(widths))
        print("\nMeasured lane width: median %.1f px (min %.0f, max %.0f)"
              % (med, min(widths), max(widths)))
        print("Put this in config.py:")
        print("    expected_lane_width: int = %d" % int(round(med)))
        print("    lane_width_tolerance: float = 0.60")
        print("\nThe tolerance gate then accepts %d to %d px and rejects "
              "anything else,\nwhich is what stops the detector locking onto "
              "the edge of the paper." % (int(med * 0.4), int(med * 1.6)))


# ======================================================================
def mode_scale(cfg, known_mm, out_path):
    """Click two points a known distance apart to get pixels per centimetre."""
    src = cam.create_source(cfg.camera)
    print("Point the camera at the calibration sheet.")
    print("SPACE freezes the frame, then click the two reference points.")
    print("The default reference is %.0f mm. q quits." % known_mm)

    points = []
    frozen = None

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and frozen is not None:
            points.append((x / 2.0, y / 2.0))

    cv2.namedWindow("calibration")
    cv2.setMouseCallback("calibration", on_mouse)

    px_per_mm = None
    try:
        while True:
            if frozen is None:
                ts, frame = src.read()
                if frame is None:
                    break
                frame = cam.orient(frame, cfg.camera)
            else:
                frame = frozen
            view = cv2.resize(frame, None, fx=2.0, fy=2.0,
                              interpolation=cv2.INTER_NEAREST)
            for p in points:
                cv2.circle(view, (int(p[0] * 2), int(p[1] * 2)), 5, (60, 200, 250), -1)
            if len(points) >= 2:
                cv2.line(view, (int(points[0][0] * 2), int(points[0][1] * 2)),
                         (int(points[1][0] * 2), int(points[1][1] * 2)),
                         (120, 230, 140), 2)
                dist = float(np.hypot(points[1][0] - points[0][0],
                                      points[1][1] - points[0][1]))
                px_per_mm = dist / known_mm
                msg = "%.1f px over %.0f mm  ->  %.3f px/mm = %.2f px/cm" % (
                    dist, known_mm, px_per_mm, px_per_mm * 10)
                cv2.putText(view, msg, (8, 24), FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(view, msg, (8, 24), FONT, 0.45, OK, 1, cv2.LINE_AA)
                cv2.putText(view, "s saves, r resets", (8, 44), FONT, 0.4,
                            (200, 200, 200), 1, cv2.LINE_AA)
            else:
                hint = "SPACE to freeze, then click two points" if frozen is None \
                    else "click point %d of 2" % (len(points) + 1)
                cv2.putText(view, hint, (8, 24), FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(view, hint, (8, 24), FONT, 0.45, (60, 200, 250), 1,
                            cv2.LINE_AA)
            cv2.imshow("calibration", view)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord(" "):
                frozen = frame.copy() if frozen is None else None
                points.clear()
            if k == ord("r"):
                points.clear()
            if k == ord("s") and px_per_mm:
                data = {
                    "px_per_mm": round(px_per_mm, 4),
                    "px_per_cm": round(px_per_mm * 10, 3),
                    "reference_mm": known_mm,
                    "resolution": [cfg.camera.width, cfg.camera.height],
                    "measured": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "note": "valid only for this camera height and angle",
                }
                with open(out_path, "w") as fh:
                    json.dump(data, fh, indent=2)
                print("\nSaved %s" % out_path)
                print(json.dumps(data, indent=2))
                print("\nUse it with:  python3 report.py --run <run> --px-per-cm %.2f"
                      % data["px_per_cm"])
                break
    finally:
        src.release()
        cv2.destroyAllWindows()


# ======================================================================
def mode_check(cfg, seconds=30):
    """Automated health check. Prints a pass/fail list."""
    results = []

    def record(label, ok, detail=""):
        results.append((label, ok, detail))
        mark = "PASS" if ok is True else ("WARN" if ok is None else "FAIL")
        print("  %-4s  %-34s %s" % (mark, label, detail))

    print("\nSystem health check (%d s)\n" % seconds)

    # --- platform
    temp = cpu_temperature()
    record("CPU temperature readable", True if temp is not None else None,
           "%.1f C" % temp if temp else "not available (not a Raspberry Pi?)")
    if temp is not None:
        record("CPU below throttling threshold", temp < 70.0,
               "%.1f C (throttles at 80 C)" % temp)
    value, flags = throttled_flags()
    if value is not None:
        record("No throttling or under-voltage", not flags,
               ", ".join(flags) if flags else "clean (0x0)")
    clock = cpu_clock_mhz()
    if clock:
        record("ARM clock at full speed", clock > 1300,
               "%.0f MHz" % clock)

    try:
        import RPi.GPIO  # noqa: F401
        record("RPi.GPIO available", True, "motor control possible")
    except Exception as exc:
        record("RPi.GPIO available", None, "mock driver will be used (%s)"
               % type(exc).__name__)

    # --- camera and pipeline
    try:
        src = cam.create_source(cfg.camera)
    except Exception as exc:
        record("Camera opens", False, str(exc))
        return results
    record("Camera opens", True, src.name)

    pre = Preprocessor(cfg.vision)
    det = create_detector(cfg.algorithm, cfg.vision, pre)

    lat, fps_l, det_l, conf_l, temps = [], [], [], [], []
    t_end = time.time() + seconds
    prev = None
    try:
        while time.time() < t_end:
            t_cap, frame = src.read()
            if frame is None:
                break
            frame = cam.orient(frame, cfg.camera)
            p = pre.process(frame)
            r = det.detect(p, cfg.camera.width)
            lat.append((time.perf_counter() - t_cap) * 1000)
            now = time.perf_counter()
            if prev:
                fps_l.append(1.0 / max(1e-6, now - prev))
            prev = now
            det_l.append(1 if r.detected else 0)
            conf_l.append(r.confidence)
            if len(det_l) % 40 == 0:
                t = cpu_temperature()
                if t:
                    temps.append(t)
    finally:
        src.release()

    if not fps_l:
        record("Frames captured", False, "no frames")
        return results

    mean_fps = float(np.mean(fps_l))
    mean_lat = float(np.mean(lat))
    rate = float(np.mean(det_l))

    record("Frames captured", True, "%d frames" % len(det_l))
    record("Frame rate >= 15 FPS (H4)", mean_fps >= 15.0,
           "%.1f FPS (min %.1f)" % (mean_fps, min(fps_l)))
    record("Latency < 80 ms (H2)", mean_lat < 80.0,
           "mean %.1f ms, p95 %.1f ms" % (mean_lat, np.percentile(lat, 95)))
    record("Lane detected in >95 % of frames", rate > 0.95,
           "%.1f %% (mean confidence %.2f)" % (100 * rate, np.mean(conf_l)))
    if len(temps) > 1:
        rise = temps[-1] - temps[0]
        record("Temperature stable", rise < 8.0,
               "%.1f C -> %.1f C" % (temps[0], temps[-1]))

    failed = [r for r in results if r[1] is False]
    print("\n  %d checks, %d failed" % (len(results), len(failed)))
    hints = {
            "Frame rate >= 15 FPS (H4)":
                "run main.py with --no-graphs, or set dashboard.update_every = 2",
            "Latency < 80 ms (H2)":
                "keep camera.threaded = True and lower the capture resolution",
            "Lane detected in >95 % of frames":
                "run --mode place and fix the camera aim, or switch threshold mode",
            "No throttling or under-voltage":
                "use a 2.5 A supply for the Pi; do not power it from the motor pack",
            "CPU below throttling threshold":
                "add a heatsink, the Pi 3B+ throttles hard without one",
    }
    advice = [(l, hints[l]) for l, _, _ in failed if l in hints]
    if advice:
        print("\nWhat to do:")
        for label, hint in advice:
            print("  * %s: %s" % (label, hint))
    print()
    return results


# ======================================================================
def mode_thermal(cfg, minutes, out_dir):
    """Long run logging temperature, clock and frame rate."""
    os.makedirs(out_dir, exist_ok=True)
    src = cam.create_source(cfg.camera)
    pre = Preprocessor(cfg.vision)
    det = create_detector(cfg.algorithm, cfg.vision, pre)

    rows = []
    t0 = time.time()
    t_end = t0 + minutes * 60
    prev = None
    print("Thermal soak test for %.0f minutes. Ctrl+C to stop early." % minutes)
    try:
        while time.time() < t_end:
            t_cap, frame = src.read()
            if frame is None:
                break
            p = pre.process(cam.orient(frame, cfg.camera))
            det.detect(p, cfg.camera.width)
            now = time.perf_counter()
            fps = 1.0 / max(1e-6, now - prev) if prev else 0.0
            prev = now
            if len(rows) == 0 or time.time() - rows[-1]["t"] - t0 > 2.0:
                value, flags = throttled_flags()
                rows.append({
                    "t": time.time() - t0,
                    "temp_c": cpu_temperature() or float("nan"),
                    "clock_mhz": cpu_clock_mhz() or float("nan"),
                    "fps": fps,
                    "throttled": 1 if flags else 0,
                })
                print("\r  %5.1f min  %.1f C  %.0f MHz  %.1f FPS  %s"
                      % (rows[-1]["t"] / 60, rows[-1]["temp_c"],
                         rows[-1]["clock_mhz"], fps,
                         "THROTTLED" if flags else "ok"), end="")
    except KeyboardInterrupt:
        print("\n  stopped early")
    finally:
        src.release()

    if not rows:
        return
    import csv
    path = os.path.join(out_dir, "thermal.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("\n  wrote %s" % path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    t = [r["t"] / 60 for r in rows]
    fig, ax1 = plt.subplots(figsize=(7, 3.6))
    ax1.plot(t, [r["temp_c"] for r in rows], color="#c0504d", label="CPU temperature")
    ax1.axhline(80, color="#c0504d", ls="--", lw=0.9, label="throttle threshold")
    ax1.set_xlabel("time (min)")
    ax1.set_ylabel("temperature (C)", color="#c0504d")
    ax2 = ax1.twinx()
    ax2.plot(t, [r["fps"] for r in rows], color="#1f4e79", lw=0.9, label="frame rate")
    ax2.set_ylabel("frame rate (FPS)", color="#1f4e79")
    ax1.set_title("Thermal behaviour during sustained operation")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, "thermal.%s" % ext), dpi=300)
    plt.close(fig)
    print("  wrote %s/thermal.png" % out_dir)


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="Calibration and health checks")
    ap.add_argument("--mode", required=True,
                    choices=["place", "scale", "check", "thermal"])
    ap.add_argument("--config")
    ap.add_argument("--known-mm", type=float, default=200.0,
                    help="reference distance for --mode scale")
    ap.add_argument("--seconds", type=int, default=30,
                    help="duration of --mode check")
    ap.add_argument("--minutes", type=float, default=10.0,
                    help="duration of --mode thermal")
    ap.add_argument("--out", default="results/calibration")
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()

    if args.mode == "place":
        mode_place(cfg)
    elif args.mode == "scale":
        os.makedirs(args.out, exist_ok=True)
        mode_scale(cfg, args.known_mm,
                   os.path.join(args.out, "calibration.json"))
    elif args.mode == "check":
        mode_check(cfg, args.seconds)
    else:
        mode_thermal(cfg, args.minutes, args.out)


if __name__ == "__main__":
    main()
