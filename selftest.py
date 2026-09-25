#!/usr/bin/env python3
"""
selftest.py -- Verifies the whole software stack without a camera or motors.

Run this first, on your laptop or on the Pi:

    python3 selftest.py

It checks that:
  * both detectors recover a known lane offset from synthetic frames
  * all three thresholding modes work
  * the PID reduces a constant error and does not wind up
  * the deadzone, output limits and anti-windup behave as specified
  * the speed mixer produces the correct differential steering direction
  * the motor driver falls back to the mock cleanly off-Pi
  * the metrics recorder produces a complete summary

Any FAIL line means that file needs fixing before you touch the hardware.
"""

import sys

import numpy as np

import camera as cam
from config import Config
from controller import PIDController, SpeedMixer, BangBangController
from detectors import create_detector
from metrics import MetricsRecorder
from motors import create_driver
from vision import Preprocessor

PASS, FAIL = 0, 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print("  PASS  %s %s" % (label, detail))
    else:
        FAIL += 1
        print("  FAIL  %s %s" % (label, detail))


# ----------------------------------------------------------------------
def test_detectors():
    print("\n[1] Lane detection")
    np.random.seed(0)          # the synthetic frames add random noise
    cfg = Config()
    # camera.synthetic_frame draws BRIGHT lane lines on a DARK background,
    # the opposite polarity to a black-tape-on-white-paper track. The test
    # sets the flag explicitly so it stays valid whatever the config default
    # is. Note what a wrong flag costs: it produces a large, almost constant
    # lane-centre error while still reporting high confidence, because the
    # detector locks onto the background instead of the markings.
    cfg.vision.invert_binary = False
    w, h = cfg.camera.width, cfg.camera.height
    for mode in ("fixed", "otsu", "adaptive"):
        cfg.vision.threshold_mode = mode
        pre = Preprocessor(cfg.vision)
        for algo in ("histogram", "canny_hough"):
            det = create_detector(algo, cfg.vision, pre)
            errors = []
            ok = True
            for shift in (-40, -20, 0, 20, 40):
                frame = cam.synthetic_frame(w, h, offset=shift, noise=4)
                p = pre.process(frame)
                res = det.detect(p, w)
                if not res.detected:
                    ok = False
                    break
                errors.append(abs((res.lane_center - w / 2) - shift))
            mean_err = float(np.mean(errors)) if errors else 999
            check("%-12s / %-8s" % (algo, mode), ok and mean_err < 12,
                  "mean centre error %.1f px" % mean_err)


def test_pid():
    print("\n[2] PID controller")
    cfg = Config()
    cfg.control.deadzone_px = 10.0
    pid = PIDController(cfg.control)

    out = pid.update(5.0, 160.0, 1 / 15.0)
    check("deadzone suppresses small error", abs(out) < 1e-9, "out=%.4f" % out)

    pid.reset()
    outs = [pid.update(60.0, 160.0, 1 / 15.0) for _ in range(30)]
    check("positive error -> positive steer", outs[0] > 0, "first=%.3f" % outs[0])
    check("output respects the limit",
          max(abs(o) for o in outs) <= cfg.control.output_limit + 1e-9)
    check("integral is clamped (anti-windup)",
          abs(pid.integral * pid.ki) <= cfg.control.integral_limit + 1e-6,
          "I=%.3f" % (pid.integral * pid.ki))

    pid.reset()
    err = 60.0
    for _ in range(120):                      # crude first-order plant
        u = pid.update(err, 160.0, 1 / 15.0)
        err -= 25.0 * u * (1 / 15.0) * 4.0
    check("closed loop converges", abs(err) < 15.0, "residual=%.1f px" % err)

    pid.reset()
    # Inspect the D term itself rather than the controller output: with a high
    # Kp the output saturates at the limit and the two samples become equal,
    # which says nothing about the derivative.
    d_terms = []
    for e in (0, 0, 80, 80, 80):
        pid.update(e, 160.0, 1 / 15.0)
        d_terms.append(pid.terms[2])
    check("derivative reacts to a step then decays",
          d_terms[2] > 0 and d_terms[2] > d_terms[4],
          "%.4f -> %.4f" % (d_terms[2], d_terms[4]))

    bb = BangBangController(cfg.control)
    check("bang-bang thresholds",
          bb.update(-50, 160, 0.05) == -1.0 and
          bb.update(0, 160, 0.05) == 0.0 and
          bb.update(50, 160, 0.05) == 1.0)


def test_mixer():
    print("\n[3] Speed mixing")
    cfg = Config()
    mixer = SpeedMixer(cfg.control)
    l0, r0 = mixer.mix(0.0)
    check("straight -> equal wheels", abs(l0 - r0) < 1e-9, "%.2f/%.2f" % (l0, r0))
    lr, rr = mixer.mix(0.8)
    check("steer right -> left wheel faster", lr > rr, "%.2f/%.2f" % (lr, rr))
    ll, rl = mixer.mix(-0.8)
    check("steer left -> right wheel faster", rl > ll, "%.2f/%.2f" % (ll, rl))
    check("wheel speeds stay in range",
          all(cfg.control.min_speed - 1e-9 <= v <= cfg.control.max_speed + 1e-9
              for v in (lr, rr, ll, rl)))


def test_motors():
    print("\n[4] Motor driver")
    cfg = Config()
    drv = create_driver(cfg.motors)
    drv.drive(0.0, 0.0)
    check("deadband -> zero duty", drv.left_duty == 0.0)
    drv.drive(0.5, 0.5)
    check("small request lifted above motor_min_duty",
          drv.left_duty >= cfg.motors.motor_min_duty,
          "duty=%.0f%%" % drv.left_duty)
    drv.drive(1.0, 1.0)
    check("full request -> 100 %", abs(drv.left_duty - 100.0) < 1e-6)
    drv.stop()
    check("stop clears both channels",
          drv.left_duty == 0.0 and drv.right_duty == 0.0)
    drv.cleanup()


def test_metrics():
    print("\n[5] Metrics")
    cfg = Config()
    cfg.logging.enabled = False
    rec = MetricsRecorder(cfg.logging, cfg)
    for i in range(40):
        rec.add({"detected": 1, "error_px": 10 * (-1) ** i,
                 "latency_ms": 55 + i % 5, "loop_ms": 60.0,
                 "confidence": 0.8, "t_blur": 4.0})
    s = rec.summary()
    check("frame count", s["frames"] == 40)
    check("instantaneous fps computed", 16 < s["fps_instantaneous_mean"] < 17,
          "%.2f" % s["fps_instantaneous_mean"])
    check("throughput reported", s["fps_throughput"] > 0,
          "%.2f fps" % s["fps_throughput"])
    check("oscillation counted", s["oscillation_zero_crossings"] == 39,
          str(s["oscillation_zero_crossings"]))
    check("H2 flag", s["H2_latency_under_80ms"] is True)
    rec.close()


def test_pipeline():
    print("\n[6] End-to-end loop (mock hardware)")
    cfg = Config()
    cfg.vision.invert_binary = False       # synthetic frames, see test_detectors
    cfg.camera.source = "video"
    pre = Preprocessor(cfg.vision)
    det = create_detector("histogram", cfg.vision, pre)
    pid = PIDController(cfg.control)
    mixer = SpeedMixer(cfg.control)
    drv = create_driver(cfg.motors, force_mock=True)

    ok = True
    for i in range(60):
        frame = cam.synthetic_frame(cfg.camera.width, cfg.camera.height,
                                    offset=35 * np.sin(i / 8.0), noise=8)
        p = pre.process(frame)
        res = det.detect(p, cfg.camera.width)
        if not res.detected:
            ok = False
            continue
        steer = pid.update(res.lane_center - cfg.camera.width / 2,
                           cfg.camera.width / 2, 1 / 15.0)
        l, r = mixer.mix(steer)
        drv.drive(l, r)
    check("60 frames processed without error", ok)
    check("motors received a duty", drv.left_duty > 0)
    drv.cleanup()


def main():
    print("=" * 60)
    print("Lane-following system self-test")
    print("=" * 60)
    test_detectors()
    test_pid()
    test_mixer()
    test_motors()
    test_metrics()
    test_pipeline()
    print("\n" + "=" * 60)
    print("  %d passed, %d failed" % (PASS, FAIL))
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
