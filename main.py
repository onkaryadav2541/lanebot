#!/usr/bin/env python3
"""
main.py -- Autonomous lane-following vehicle, main control loop.

    Capture -> Preprocess -> Detect -> PID -> Motor mix -> Drive -> Telemetry

Run from Thonny (green Run button) or from a terminal:

    python3 main.py                          # histogram + PID, motors armed on 's'
    python3 main.py --algo canny_hough       # Algorithm A
    python3 main.py --controller bangbang    # research-paper baseline
    python3 main.py --no-motors              # vision only, safe on the bench
    python3 main.py --headless               # over SSH without a display
    python3 main.py --source video --video road.mp4   # replay a recording

Keyboard (dashboard window must have focus):
    q / ESC  quit               s      arm / disarm the motors (SPACE = e-stop)
    a        switch algorithm   c      switch controller (PID <-> bang-bang)
    t        cycle threshold mode (fixed / otsu / adaptive)   -> hypothesis H3
    1 / 2    Kp  -/+            3 / 4  Ki -/+        5 / 6  Kd -/+
    [ / ]    base speed -/+     d      toggle debug windows
    r        reset the PID state and the trail

Author: Onkar Yadav (Matr. 100002351), SRH Hochschule Heidelberg
"""

import argparse
import os
import signal
import sys
import time

import cv2
import numpy as np

import camera as cam
from config import Config
from controller import LoopTimer, SpeedMixer, create_controller
from dashboard import Dashboard, debug_windows
from detectors import create_detector
from metrics import MetricsRecorder, print_summary
from motors import create_driver
from vision import Preprocessor

THRESHOLD_MODES = ["fixed", "otsu", "adaptive"]


# ----------------------------------------------------------------------
def build_config(args):
    cfg = Config.load(args.config) if args.config else Config()

    if args.algo:
        cfg.algorithm = args.algo
    if args.controller:
        cfg.control.controller = args.controller
    if args.source:
        cfg.camera.source = args.source
    if args.video:
        cfg.camera.source = "video"
        cfg.camera.video_path = args.video
    if args.threshold:
        cfg.vision.threshold_mode = args.threshold
    if args.speed is not None:
        cfg.control.base_speed = args.speed
    if args.kp is not None:
        cfg.control.kp = args.kp
    if args.ki is not None:
        cfg.control.ki = args.ki
    if args.kd is not None:
        cfg.control.kd = args.kd
    if args.pwm_mode:
        cfg.motors.pwm_mode = args.pwm_mode
    if args.no_motors:
        cfg.motors.enabled = False
    if args.headless:
        cfg.dashboard.headless = True
    if args.no_log:
        cfg.logging.enabled = False
    if args.record:
        cfg.logging.record_video = True
    if args.no_graphs:
        cfg.dashboard.show_graphs = False
        cfg.dashboard.show_histogram_strip = False
    if args.scale:
        cfg.dashboard.display_scale = args.scale
    if args.every:
        cfg.dashboard.update_every = max(1, args.every)
    if args.name:
        cfg.run_name = args.name
    else:
        cfg.run_name = "%s_%s" % (cfg.algorithm, cfg.control.controller)
    return cfg


# ----------------------------------------------------------------------
class LaneFollower:
    def __init__(self, cfg):
        self.cfg = cfg
        self.running = True
        self.motors_armed = False
        self.auto_arm_at = None

        self.source = cam.create_source(cfg.camera, loop=False, realtime=True)
        self.pre = Preprocessor(cfg.vision)
        self.detector = create_detector(cfg.algorithm, cfg.vision, self.pre)
        self.controller = create_controller(cfg.control)
        self.mixer = SpeedMixer(cfg.control)
        self.motors = create_driver(cfg.motors, force_mock=not cfg.motors.enabled)
        self.metrics = MetricsRecorder(cfg.logging, cfg)
        self.dash = Dashboard(cfg.dashboard, cfg.camera.width, cfg.camera.height,
                              deadzone_px=cfg.control.deadzone_px)
        self.timer = LoopTimer()

        self.smoothed_error = 0.0
        self.last_steer = 0.0
        self.lost_frames = 0
        self.marker_state = "NORMAL"      # NORMAL | TURN | CLEAR | COOLDOWN
        self.marker_t = 0.0
        self.marker_hits = 0
        self.marker_count = 0
        self.writer = None

        if cfg.logging.record_video and self.metrics.dir:
            path = os.path.join(self.metrics.dir, "raw.avi")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.writer = cv2.VideoWriter(
                path, fourcc, cfg.logging.record_fps,
                (cfg.camera.width, cfg.camera.height))

        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    def _on_signal(self, *_):
        self.running = False

    # ------------------------------------------------------------------
    def run(self):
        cfg = self.cfg
        half_width = cfg.camera.width / 2.0
        target = cfg.camera.width / 2.0
        print(self._banner())

        while self.running:
            t_cap, frame = self.source.read()
            if frame is None:
                print("[main] frame source exhausted")
                break
            frame = cam.orient(frame, cfg.camera)
            if self.writer is not None:
                self.writer.write(frame)

            loop_dt = self.timer.tick()

            # Auto-arm. A headless run has no window, so there is no key press
            # to arm with; without this the vehicle would log a full run
            # without ever moving.
            if (self.auto_arm_at is not None and not self.motors_armed
                    and time.perf_counter() >= self.auto_arm_at):
                self.motors_armed = True
                self.auto_arm_at = None
                self.controller.reset()
                print("[main] motors ARMED (auto)")

            # -------- perception
            pre = self.pre.process(frame)
            result = self.detector.detect(pre, cfg.camera.width)

            # -------- error
            if result.detected:
                error_px = result.lane_center - target
                self.lost_frames = 0
            else:
                self.lost_frames += 1
                error_px = 0.0

            # -------- stop marker
            # Handled ahead of the controller: while a marker manoeuvre is in
            # progress the vehicle is executing a fixed sequence and the lane
            # error is not meaningful.
            if cfg.control.marker_enabled:
                handled, m_left, m_right, m_label = self._marker_step(
                    result.line_width if result.detected else 0.0, loop_dt)
                if handled:
                    if not self.motors_armed:
                        self.motors.stop()
                        m_left = m_right = 0.0
                        m_label = "STOP"
                    else:
                        self.motors.drive(m_left, m_right)
                    self._finish_iteration(
                        frame, pre, result, error_px, 0.0, m_left, m_right,
                        t_cap, loop_dt, 0.0, m_label)
                    continue

            # -------- control
            t0 = time.perf_counter()
            if hasattr(self.controller, "wheels"):
                # stop-and-turn drives the wheels directly: a pivot is not a
                # steering command that the speed mixer could express
                left, right, phase = self.controller.wheels(
                    error_px, result.detected, loop_dt, result)
                steer = self.controller.last_sign if phase.startswith("PIVOT") else 0.0
                control_ms = (time.perf_counter() - t0) * 1000
                stop_now = not self.motors_armed
                if stop_now:
                    self.motors.stop()
                    left = right = 0.0
                else:
                    self.motors.drive(left, right)
                self._finish_iteration(frame, pre, result, error_px, steer,
                                       left, right, t_cap, loop_dt, control_ms,
                                       phase if not stop_now else "STOP")
                continue

            if result.detected:
                steer = self.controller.update(error_px, half_width, loop_dt)
                self.last_steer = steer
            elif self.lost_frames <= cfg.control.lost_lane_hold_frames:
                steer = self.last_steer                    # coast on last command
            else:
                steer = 0.0
                self.controller.reset()
            control_ms = (time.perf_counter() - t0) * 1000

            left, right = self.mixer.mix(steer)
            stop_now = (not self.motors_armed or
                        self.lost_frames > cfg.control.lost_lane_stop_frames)
            if stop_now:
                self.motors.stop()
                left = right = 0.0
            else:
                self.motors.drive(left, right)

            t_cmd = time.perf_counter()
            latency_ms = (t_cmd - t_cap) * 1000.0

            # -------- telemetry
            a = cfg.control.smoothing_alpha
            self.smoothed_error = (1 - a) * self.smoothed_error + a * error_px

            terms = getattr(self.controller, "terms", (0.0, 0.0, 0.0))
            command = self.controller.last_command if result.detected else "HOLD"
            if stop_now:
                command = "STOP"

            render_ms = 0.0
            if cfg.dashboard.enabled and not cfg.dashboard.headless:
                if self.metrics.frame % max(1, cfg.dashboard.update_every) == 0:
                    t0 = time.perf_counter()
                    state = {
                        "algo": self.detector.name,
                        "controller": self.controller.name,
                        "error_px": error_px if result.detected else float("nan"),
                        "smoothed_error": self.smoothed_error,
                        "command": command,
                        "terms": terms,
                        "fps": self.metrics.fps,
                        "latency_ms": self.metrics.latency,
                        "left_duty": self.motors.left_duty,
                        "right_duty": self.motors.right_duty,
                        "motors_armed": self.motors_armed,
                        "deadzone_px": (cfg.control.deadzone_px
                                        if self.controller.name == "pid"
                                        else cfg.control.bangbang_deadzone_px),
                    }
                    canvas = self.dash.render(frame, pre, result, state)
                    cv2.imshow("Lane Following - Telemetry", canvas)
                    if cfg.dashboard.show_debug_windows:
                        debug_windows(pre, result)
                    render_ms = (time.perf_counter() - t0) * 1000
                    if not self._handle_keys():
                        break

            rec = {
                "algorithm": self.detector.name,
                "controller": self.controller.name,
                "armed": int(self.motors_armed and not stop_now),
                "detected": int(result.detected),
                "inferred": int(result.inferred),
                "confidence": result.confidence,
                "lane_center": result.lane_center,
                "left_x": result.left_x if result.left_x is not None else "",
                "right_x": result.right_x if result.right_x is not None else "",
                "error_px": error_px,
                "smoothed_error_px": self.smoothed_error,
                "steer": steer,
                "command": command,
                "p_term": terms[0], "i_term": terms[1], "d_term": terms[2],
                "left_duty": self.motors.left_duty,
                "right_duty": self.motors.right_duty,
                "latency_ms": latency_ms,
                "loop_ms": loop_dt * 1000.0,
                "fps": (1.0 / loop_dt) if loop_dt > 0 else 0.0,
                "t_detect": result.detect_ms,
                "t_control": control_ms,
                "t_render": render_ms,
            }
            for k, v in pre.timings.items():
                rec["t_" + k] = v
            self.metrics.add(rec)

        self.shutdown()

    # ------------------------------------------------------------------
    def _marker_step(self, line_width, dt):
        """
        Stop-marker state machine.

        Returns (handled, left, right, label). When handled is True the
        controller is bypassed for this frame because a fixed manoeuvre is
        running.
        """
        c = self.cfg.control
        self.marker_t += dt

        if self.marker_state == "TURN":
            if self.marker_t >= c.marker_turn_s:
                self.marker_state = "CLEAR"
                self.marker_t = 0.0
            else:
                spd = c.pivot_speed
                if c.marker_turn_direction == "left":
                    return True, -spd, spd, "MARKER L"
                return True, spd, -spd, "MARKER R"

        if self.marker_state == "CLEAR":
            if self.marker_t >= c.marker_clear_s:
                self.marker_state = "COOLDOWN"
                self.marker_t = 0.0
            else:
                return True, c.drive_speed, c.drive_speed, "CLEARING"

        if self.marker_state == "COOLDOWN":
            if self.marker_t >= c.marker_cooldown_s:
                self.marker_state = "NORMAL"
                self.marker_hits = 0
            return False, 0.0, 0.0, ""

        # --- NORMAL: watch for a widened section of lane marking ---------
        if line_width >= c.marker_line_width_px:
            self.marker_hits += 1
            if self.marker_hits >= c.marker_frames:
                self.marker_state = "TURN"
                self.marker_t = 0.0
                self.marker_hits = 0
                self.marker_count += 1
                print("[main] marker %d (line width %.0f px) -> turning %s "
                      "for %.2f s" % (self.marker_count, line_width,
                                      c.marker_turn_direction, c.marker_turn_s))
                return True, 0.0, 0.0, "MARKER"
        else:
            self.marker_hits = 0
        return False, 0.0, 0.0, ""

    # ------------------------------------------------------------------
    def _finish_iteration(self, frame, pre, result, error_px, steer,
                          left, right, t_cap, loop_dt, control_ms, command):
        """Telemetry, dashboard and logging for the stop-and-turn path."""
        cfg = self.cfg
        t_cmd = time.perf_counter()
        latency_ms = (t_cmd - t_cap) * 1000.0

        a = cfg.control.smoothing_alpha
        self.smoothed_error = (1 - a) * self.smoothed_error + a * error_px
        terms = getattr(self.controller, "terms", (0.0, 0.0, 0.0))

        render_ms = 0.0
        if cfg.dashboard.enabled and not cfg.dashboard.headless:
            if self.metrics.frame % max(1, cfg.dashboard.update_every) == 0:
                t0 = time.perf_counter()
                state = {
                    "algo": self.detector.name,
                    "controller": self.controller.name,
                    "error_px": error_px if result.detected else float("nan"),
                    "smoothed_error": self.smoothed_error,
                    "command": command,
                    "terms": terms,
                    "fps": self.metrics.fps,
                    "latency_ms": self.metrics.latency,
                    "left_duty": self.motors.left_duty,
                    "right_duty": self.motors.right_duty,
                    "motors_armed": self.motors_armed,
                    "deadzone_px": cfg.control.pivot_enter_px,
                }
                canvas = self.dash.render(frame, pre, result, state)
                cv2.imshow("Lane Following - Telemetry", canvas)
                if cfg.dashboard.show_debug_windows:
                    debug_windows(pre, result)
                render_ms = (time.perf_counter() - t0) * 1000
                if not self._handle_keys():
                    self.running = False

        rec = {
            "algorithm": self.detector.name,
            "controller": self.controller.name,
            "armed": int(self.motors_armed),
            "detected": int(result.detected),
            "inferred": int(result.inferred),
            "confidence": result.confidence,
            "lane_center": result.lane_center,
            "left_x": result.left_x if result.left_x is not None else "",
            "right_x": result.right_x if result.right_x is not None else "",
            "error_px": error_px,
            "smoothed_error_px": self.smoothed_error,
            "steer": steer,
            "command": command,
            "p_term": terms[0], "i_term": terms[1], "d_term": terms[2],
            "left_duty": self.motors.left_duty,
            "right_duty": self.motors.right_duty,
            "latency_ms": latency_ms,
            "loop_ms": loop_dt * 1000.0,
            "fps": (1.0 / loop_dt) if loop_dt > 0 else 0.0,
            "t_detect": result.detect_ms,
            "t_control": control_ms,
            "t_render": render_ms,
        }
        for k, v in pre.timings.items():
            rec["t_" + k] = v
        self.metrics.add(rec)

    # ------------------------------------------------------------------
    def _handle_keys(self):
        cfg = self.cfg
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            return True
        ch = chr(key) if 32 <= key < 127 else ""

        if key in (27,) or ch == "q":
            return False
        if ch == " ":
            self.motors_armed = False
            self.motors.stop()
            print("[main] EMERGENCY STOP")
        elif ch == "s":
            self.motors_armed = not self.motors_armed
            if not self.motors_armed:
                self.motors.stop()
            self.controller.reset()
            print("[main] motors %s" % ("ARMED" if self.motors_armed else "disarmed"))
        elif ch == "a":
            new = "canny_hough" if self.detector.name == "histogram" else "histogram"
            self.detector = create_detector(new, cfg.vision, self.pre)
            self.controller.reset()
            print("[main] algorithm -> %s" % new)
        elif ch == "c":
            cfg.control.controller = ("bangbang" if self.controller.name == "pid"
                                      else "pid")
            self.controller = create_controller(cfg.control)
            print("[main] controller -> %s" % cfg.control.controller)
        elif ch == "t":
            i = THRESHOLD_MODES.index(cfg.vision.threshold_mode)
            cfg.vision.threshold_mode = THRESHOLD_MODES[(i + 1) % len(THRESHOLD_MODES)]
            print("[main] threshold mode -> %s" % cfg.vision.threshold_mode)
        elif ch in "123456" and hasattr(self.controller, "set_gains"):
            step = {"1": ("kp", -0.05), "2": ("kp", +0.05),
                    "3": ("ki", -0.02), "4": ("ki", +0.02),
                    "5": ("kd", -0.01), "6": ("kd", +0.01)}[ch]
            gain = getattr(self.controller, step[0]) + step[1]
            self.controller.set_gains(**{step[0]: gain})
            print("[main] Kp=%.3f Ki=%.3f Kd=%.3f" %
                  (self.controller.kp, self.controller.ki, self.controller.kd))
        elif ch == "[":
            cfg.control.base_speed = max(0.0, cfg.control.base_speed - 0.05)
            print("[main] base speed %.2f" % cfg.control.base_speed)
        elif ch == "]":
            cfg.control.base_speed = min(1.0, cfg.control.base_speed + 0.05)
            print("[main] base speed %.2f" % cfg.control.base_speed)
        elif ch == "d":
            cfg.dashboard.show_debug_windows = not cfg.dashboard.show_debug_windows
            if not cfg.dashboard.show_debug_windows:
                for w in ("1 gray", "2 blurred", "3 binary", "4 canny", "5 histogram"):
                    try:
                        cv2.destroyWindow(w)
                    except cv2.error:
                        pass
        elif ch == "m":
            cfg.control.marker_turn_direction = (
                "left" if cfg.control.marker_turn_direction == "right"
                else "right")
            print("[main] marker turn direction -> %s"
                  % cfg.control.marker_turn_direction)
        elif ch == "," :
            cfg.control.marker_turn_s = max(0.1, cfg.control.marker_turn_s - 0.1)
            print("[main] marker turn %.2f s" % cfg.control.marker_turn_s)
        elif ch == ".":
            cfg.control.marker_turn_s += 0.1
            print("[main] marker turn %.2f s" % cfg.control.marker_turn_s)
        elif ch == "r":
            self.controller.reset()
            self.dash.trail.clear()
            self.smoothed_error = 0.0
        return True

    # ------------------------------------------------------------------
    def _banner(self):
        c = self.cfg
        return (
            "\n" + "=" * 58 +
            "\n Autonomous Lane Following - Raspberry Pi 3B+"
            "\n" + "=" * 58 +
            "\n  source      : %s" % self.source.name +
            "\n  resolution  : %dx%d" % (c.camera.width, c.camera.height) +
            "\n  algorithm   : %s" % self.detector.name +
            "\n  controller  : %s (Kp=%.2f Ki=%.2f Kd=%.2f)" % (
                self.controller.name, c.control.kp, c.control.ki, c.control.kd) +
            "\n  threshold   : %s" % c.vision.threshold_mode +
            "\n  motors      : %s (%s)" % (
                "real L298N" if self.motors.available else "MOCK / disabled",
                c.motors.pwm_mode) +
            "\n  logging to  : %s" % (self.metrics.dir or "disabled") +
            "\n" + "=" * 58 +
            "\n  press 's' to arm the motors, SPACE for e-stop, 'q' to quit\n")

    # ------------------------------------------------------------------
    def shutdown(self):
        try:
            self.motors.stop()
            self.motors.cleanup()
        except Exception as exc:
            print("[main] motor cleanup: %s" % exc)
        if self.writer is not None:
            self.writer.release()
        self.source.release()
        cv2.destroyAllWindows()
        summary = self.metrics.close(extra={
            "algorithm": self.detector.name,
            "controller": self.controller.name,
            "kp": getattr(self.controller, "kp", None),
            "ki": getattr(self.controller, "ki", None),
            "kd": getattr(self.controller, "kd", None),
            "threshold_mode": self.cfg.vision.threshold_mode,
        })
        print_summary(summary)
        if self.metrics.dir:
            print("Results written to: %s\n" % self.metrics.dir)


# ----------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Autonomous lane-following vehicle")
    p.add_argument("--config", help="JSON config file to load")
    p.add_argument("--algo", choices=["histogram", "canny_hough"])
    p.add_argument("--controller",
                   choices=["pid", "bangbang", "stop_and_turn"])
    p.add_argument("--source", choices=["picamera", "usb", "video"])
    p.add_argument("--video", help="path to a recorded clip")
    p.add_argument("--threshold", choices=THRESHOLD_MODES)
    p.add_argument("--speed", type=float, help="base speed 0..1")
    p.add_argument("--kp", type=float)
    p.add_argument("--ki", type=float)
    p.add_argument("--kd", type=float)
    p.add_argument("--pwm-mode", dest="pwm_mode",
                   choices=["enable_pwm", "direction_pwm", "bangbang"])
    p.add_argument("--no-motors", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--record", action="store_true", help="record raw video")
    p.add_argument("--arm", type=float, nargs="?", const=3.0, default=None,
                   metavar="SECONDS",
                   help="arm the motors automatically after this many seconds "
                        "(default 3). Required for --headless runs, where "
                        "there is no window to press 's' in.")
    p.add_argument("--no-graphs", action="store_true",
                   help="disable the live charts (saves a few ms per frame)")
    p.add_argument("--scale", type=float,
                   help="dashboard zoom factor (1.0 is much cheaper than 2.0)")
    p.add_argument("--every", type=int,
                   help="render the dashboard only every Nth frame")
    p.add_argument("--name", help="run name used for the results folder")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = build_config(args)
    app = LaneFollower(cfg)
    if args.arm is not None:
        app.auto_arm_at = time.perf_counter() + args.arm
        print("\n  MOTORS WILL ARM IN %.0f SECONDS - place the vehicle now\n"
              % args.arm)
    if args.max_frames:
        original = app.metrics.add

        def limited(rec):
            original(rec)
            if app.metrics.frame >= args.max_frames:
                app.running = False
        app.metrics.add = limited
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
