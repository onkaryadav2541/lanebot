#!/usr/bin/env python3
"""
test_hardware.py -- Bring-up checks. Run this BEFORE main.py.

Lift the car onto a box so the wheels spin freely, then:

    python3 test_hardware.py --motors     # each channel, forward and reverse
    python3 test_hardware.py --ramp       # duty sweep: find motor_min_duty
    python3 test_hardware.py --camera     # 100 frames, reports real capture fps
    python3 test_hardware.py --vision     # live threshold/Canny preview
    python3 test_hardware.py --all

What to look for:

  * "left forward" must spin BOTH left wheels forward. If a side spins
    backwards, set invert_left / invert_right in config.py. If left and right
    are swapped, set swap_sides.
  * In the ramp test, note the lowest duty at which the wheels actually start
    turning under the weight of the car and put that number into
    MotorConfig.motor_min_duty.
  * If nothing moves at all: check the ENA/ENB jumpers, the common ground
    between the battery pack and the Pi, and that the 6 V pack is charged.
    A Pi that reboots when the motors start means the motor supply is
    back-feeding the Pi or the pack is sagging.
"""

import argparse
import time

import cv2
import numpy as np

import camera as cam
from config import Config
from motors import create_driver
from vision import Preprocessor


def test_motors(cfg):
    drv = create_driver(cfg.motors)
    if not drv.available:
        print("!! Running with the MOCK driver -- no GPIO available.")
    seq = [
        ("left forward", 0.8, 0.0), ("left reverse", -0.8, 0.0),
        ("right forward", 0.0, 0.8), ("right reverse", 0.0, -0.8),
        ("both forward", 0.8, 0.8), ("pivot right", 0.8, -0.8),
        ("pivot left", -0.8, 0.8),
    ]
    try:
        for label, l, r in seq:
            print("  %-14s  left=%+.1f right=%+.1f" % (label, l, r))
            drv.drive(l, r)
            time.sleep(1.2)
            drv.stop()
            time.sleep(0.5)
    finally:
        drv.stop()
        drv.cleanup()
    print("Motor test finished.")


def test_ramp(cfg):
    cfg.motors.motor_min_duty = 0.0        # raw sweep, no remapping
    drv = create_driver(cfg.motors)
    print("Ramping both sides. Note the duty at which the wheels start moving.")
    try:
        for pct in range(10, 101, 5):
            speed = pct / 100.0
            drv.drive(speed, speed)
            print("  duty %3d %%" % pct)
            time.sleep(0.9)
    finally:
        drv.stop()
        drv.cleanup()
    print("Put that value into MotorConfig.motor_min_duty (add ~5 % margin).")


def test_camera(cfg, n=100):
    src = cam.create_source(cfg.camera)
    print("Source: %s" % src.name)
    t0 = time.perf_counter()
    got = 0
    try:
        for _ in range(n):
            ts, frame = src.read()
            if frame is None:
                break
            got += 1
    finally:
        src.release()
    dt = time.perf_counter() - t0
    print("  %d frames in %.2f s -> %.1f fps, mean %.1f ms per frame"
          % (got, dt, got / dt if dt else 0, 1000 * dt / max(1, got)))
    if got:
        print("  frame shape: %s dtype: %s" % (frame.shape, frame.dtype))


def test_vision(cfg):
    src = cam.create_source(cfg.camera)
    pre = Preprocessor(cfg.vision)
    print("Live preview. Keys:")
    print("  t = cycle threshold mode")
    print("  h = toggle horizontal flip     v = toggle vertical flip")
    print("  q = quit and print the flip settings to put in config.py")
    modes = ["fixed", "otsu", "adaptive"]
    try:
        while True:
            ts, frame = src.read()
            if frame is None:
                break
            frame = cam.orient(frame, cfg.camera)
            p = pre.process(frame)
            stack = np.hstack([
                cv2.cvtColor(p.gray, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(p.binary, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(p.edges, cv2.COLOR_GRAY2BGR)])
            cv2.putText(stack, "gray | binary (%s, T=%.0f) | canny"
                        % (cfg.vision.threshold_mode, p.threshold_used),
                        (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (60, 220, 250), 1, cv2.LINE_AA)
            cv2.imshow("vision check", cv2.resize(stack, None, fx=1.4, fy=1.4))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q") or k == 27:
                break
            if k == ord("t"):
                i = modes.index(cfg.vision.threshold_mode)
                cfg.vision.threshold_mode = modes[(i + 1) % 3]
                pre = Preprocessor(cfg.vision)
            if k == ord("h"):
                cfg.camera.flip_horizontal = not cfg.camera.flip_horizontal
                print("  flip_horizontal = %s" % cfg.camera.flip_horizontal)
            if k == ord("v"):
                cfg.camera.flip_vertical = not cfg.camera.flip_vertical
                print("  flip_vertical = %s" % cfg.camera.flip_vertical)
    finally:
        src.release()
        cv2.destroyAllWindows()
    print("\nPut these in config.py, class CameraConfig:")
    print("    flip_horizontal: bool = %s" % cfg.camera.flip_horizontal)
    print("    flip_vertical: bool = %s" % cfg.camera.flip_vertical)


def main():
    ap = argparse.ArgumentParser(description="Hardware bring-up tests")
    ap.add_argument("--motors", action="store_true")
    ap.add_argument("--ramp", action="store_true")
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    if args.all or args.camera:
        print("\n== camera ==")
        test_camera(cfg)
    if args.all or args.motors:
        print("\n== motors ==")
        test_motors(cfg)
    if args.ramp:
        print("\n== duty ramp ==")
        test_ramp(cfg)
    if args.all or args.vision:
        print("\n== vision ==")
        test_vision(cfg)
    if not any([args.motors, args.ramp, args.camera, args.vision, args.all]):
        ap.print_help()


if __name__ == "__main__":
    main()
