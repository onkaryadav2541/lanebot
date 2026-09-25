"""
controller.py -- Lateral control.

Two controllers with a common interface so the thesis can compare them
directly on the same hardware:

    BangBangController -- LEFT / STRAIGHT / RIGHT with a deadzone.
                          This reproduces the research-paper behaviour and
                          serves as the baseline.
    PIDController      -- continuous proportional-integral-derivative control,
                          the contribution of the thesis.

Implementation details that matter for a real vehicle and are worth writing
up in the thesis:

  * The error is normalised, e = error_px / (frame_width/2), so e is in
    [-1, 1] and the tuned gains do not change if the capture resolution
    changes.
  * Anti-windup by conditional integration: the integral only accumulates
    while the output is not saturated. Without this, a long curve winds the
    I term up and the vehicle keeps steering after the curve ends.
  * The derivative term is low-pass filtered. A raw derivative on a signal
    that is quantised to whole pixels and updated at ~15 Hz is mostly noise.
  * A continuous deadzone: instead of forcing the error to zero inside the
    band (which creates a discontinuity at the edge), the deadzone width is
    subtracted from the magnitude of the error. The command is therefore
    still continuous at the boundary and no step is injected into the D term.
  * dt is measured, not assumed. On a Pi the loop period fluctuates, and
    using a nominal dt makes Ki and Kd wrong exactly when the system is
    struggling.
"""

import time

import numpy as np


class BangBangController:
    name = "bangbang"

    def __init__(self, cfg):
        self.cfg = cfg
        self.deadzone = cfg.bangbang_deadzone_px
        self.last_command = "STRAIGHT"
        self.terms = (0.0, 0.0, 0.0)

    def reset(self):
        self.last_command = "STRAIGHT"

    def update(self, error_px, half_width, dt):
        if error_px < -self.deadzone:
            self.last_command = "LEFT"
            return -1.0
        if error_px > self.deadzone:
            self.last_command = "RIGHT"
            return 1.0
        self.last_command = "STRAIGHT"
        return 0.0


class PIDController:
    name = "pid"

    def __init__(self, cfg):
        self.cfg = cfg
        self.kp = cfg.kp
        self.ki = cfg.ki
        self.kd = cfg.kd
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.derivative = 0.0
        self.output = 0.0
        self.terms = (0.0, 0.0, 0.0)
        self._first = True

    # ------------------------------------------------------------------
    def update(self, error_px, half_width, dt):
        c = self.cfg
        if dt <= 0.0:
            dt = 1e-3
        dt = min(dt, 0.5)                     # ignore pathological stalls

        # continuous deadzone in pixel space
        dz = c.deadzone_px
        if abs(error_px) <= dz:
            eff_px = 0.0
        else:
            eff_px = error_px - np.sign(error_px) * dz

        e = float(eff_px) / float(max(1.0, half_width))

        # --- proportional
        p = self.kp * e

        # --- derivative (filtered)
        if self._first:
            raw_d = 0.0
            self._first = False
        else:
            raw_d = (e - self.prev_error) / dt
        a = c.d_filter_alpha
        self.derivative = a * raw_d + (1.0 - a) * self.derivative
        d = self.kd * self.derivative
        self.prev_error = e

        # --- integral with conditional integration (anti-windup)
        candidate = self.integral + e * dt
        candidate = float(np.clip(candidate,
                                  -c.integral_limit / max(self.ki, 1e-6),
                                  c.integral_limit / max(self.ki, 1e-6)))
        i_term = self.ki * candidate
        unsat = p + i_term + d

        if abs(unsat) <= c.output_limit or (np.sign(unsat) != np.sign(e)):
            self.integral = candidate
        i = self.ki * self.integral

        out = float(np.clip(p + i + d, -c.output_limit, c.output_limit))
        self.terms = (p, i, d)
        self.output = out
        return out

    # ------------------------------------------------------------------
    def set_gains(self, kp=None, ki=None, kd=None):
        if kp is not None:
            self.kp = max(0.0, kp)
        if ki is not None:
            self.ki = max(0.0, ki)
            self.integral = 0.0
        if kd is not None:
            self.kd = max(0.0, kd)

    @property
    def last_command(self):
        if self.output < -0.08:
            return "LEFT"
        if self.output > 0.08:
            return "RIGHT"
        return "STRAIGHT"


# ======================================================================
class SpeedMixer:
    """
    Converts a steering command in [-1, 1] into two wheel speeds in [-1, 1].

    Sign convention used throughout the project:
        error  = lane_center - image_center
        error > 0  ->  the lane runs to the right of the vehicle
                   ->  the vehicle must steer RIGHT
                   ->  the LEFT wheels must turn faster (differential steering)
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def mix(self, steer):
        c = self.cfg
        steer = float(np.clip(steer, -1.0, 1.0))
        base = c.base_speed * (1.0 - c.turn_slowdown * abs(steer))
        delta = c.steer_gain * steer
        left = base + delta
        right = base - delta
        left = float(np.clip(left, c.min_speed, c.max_speed))
        right = float(np.clip(right, c.min_speed, c.max_speed))
        return left, right


# ======================================================================
class StopAndTurnStrategy:
    """
    Discrete drive / pivot navigation for a platform with limited steering
    authority.

    A skid-steer chassis whose motors only start near full duty cannot produce
    a useful speed difference between the two sides, so it cannot follow a
    curve while driving. It can, however, rotate on the spot, because a pivot
    drives one side forward and the other backward and needs no differential.

    The strategy therefore alternates:

        DRIVE   both sides forward while the lane centre stays near the image
                centre
        PIVOT   forward motion stops and the vehicle rotates in short pulses,
                checking the error between pulses, until it is realigned

    Pulsing rather than rotating continuously matters: at these duty cycles the
    vehicle turns quickly, and a continuous pivot overshoots long before the
    next frame is processed. A short pulse followed by a pause lets the
    perception loop catch up, which makes the manoeuvre convergent instead of
    oscillatory.

    This is a deliberate trade of smoothness for feasibility: the path becomes
    a sequence of straight segments joined by stationary rotations rather than
    a continuous arc, which is worth stating plainly in the evaluation.
    """

    name = "stop_and_turn"

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.state = "DRIVE"
        self.phase_t = 0.0
        self.search_t = 0.0
        self.pulsing = False
        self.last_sign = 1.0
        self.search_sign = 1.0
        self.good_frames = 0
        self.terms = (0.0, 0.0, 0.0)
        self.last_command = "STRAIGHT"

    # ------------------------------------------------------------------
    def _is_solid(self, result):
        """
        A detection strong enough to steer by.

        During a search the vehicle sweeps across the lane, and on the way it
        catches glimpses of a single boundary. Those glimpses arrive with a
        low confidence and with the opposite boundary merely inferred, and
        acting on them makes the vehicle reverse direction and wobble instead
        of completing the sweep. A reacquisition therefore requires both
        boundaries, a real confidence, and several consecutive frames.
        """
        if result is None or not result.detected or result.inferred:
            return False
        return result.confidence >= self.cfg.reacquire_min_conf

    # ------------------------------------------------------------------
    def wheels(self, error_px, detected, dt, result=None):
        """Returns (left, right, label) with speeds in [-1, 1]."""
        c = self.cfg
        self.phase_t += dt
        solid = self._is_solid(result) if result is not None else detected

        if solid:
            self.good_frames += 1
            if abs(error_px) > 1e-6:
                self.last_sign = 1.0 if error_px > 0 else -1.0
        else:
            self.good_frames = 0

        # ---------------- SEARCH ---------------------------------------
        # Entered when the lane leaves the field of view entirely. The
        # rotation direction is locked on entry and held until the lane is
        # reacquired, so a stray single-boundary frame cannot reverse it.
        if self.state == "SEARCH":
            self.search_t += dt
            if self.good_frames >= c.reacquire_frames:
                self.state = "DRIVE"
                self.phase_t = self.search_t = 0.0
                self.last_command = "STRAIGHT"
                return c.drive_speed, c.drive_speed, "DRIVE"

            # sweep one way, then the other, widening the arc each time
            if self.search_t >= c.search_reverse_after_s:
                self.search_sign = -self.search_sign
                self.search_t = 0.0

            if self._pulse(c.search_pulse_s, c.search_pause_s):
                self.last_command = "RIGHT" if self.search_sign > 0 else "LEFT"
                spd = c.pivot_speed
                return ((spd, -spd, "SEARCH R") if self.search_sign > 0
                        else (-spd, spd, "SEARCH L"))
            self.last_command = "LOOK"
            return 0.0, 0.0, "LOOK"

        if not solid and not detected:
            self.state = "SEARCH"
            self.search_sign = self.last_sign
            self.search_t = 0.0
            self.phase_t = 0.0
            self.pulsing = True
            self.last_command = "SEARCH"
            return 0.0, 0.0, "SEARCH"

        magnitude = abs(error_px)

        # ---------------- DRIVE ----------------------------------------
        if self.state == "DRIVE":
            if magnitude > c.pivot_enter_px:
                self.state = "PIVOT"
                self.phase_t = 0.0
                self.pulsing = True
            else:
                self.terms = (0.0, 0.0, 0.0)
                self.last_command = "STRAIGHT"
                return c.drive_speed, c.drive_speed, "DRIVE"

        # ---------------- PIVOT ----------------------------------------
        if magnitude < c.pivot_exit_px and solid:
            self.state = "DRIVE"
            self.phase_t = 0.0
            self.last_command = "STRAIGHT"
            return c.drive_speed, c.drive_speed, "DRIVE"

        self.terms = (self.last_sign, 0.0, 0.0)
        if not self._pulse(c.pivot_pulse_s, c.pivot_pause_s):
            self.last_command = "LOOK"
            return 0.0, 0.0, "LOOK"

        spd = c.pivot_speed
        self.last_command = "RIGHT" if self.last_sign > 0 else "LEFT"
        return ((spd, -spd, "PIVOT R") if self.last_sign > 0
                else (-spd, spd, "PIVOT L"))

    # ------------------------------------------------------------------
    def _pulse(self, on_s, off_s):
        """Alternates between a rotation burst and a pause. True = rotate."""
        if self.pulsing:
            if self.phase_t >= on_s:
                self.pulsing = False
                self.phase_t = 0.0
        else:
            if self.phase_t >= off_s:
                self.pulsing = True
                self.phase_t = 0.0
        return self.pulsing


# ======================================================================
def create_controller(cfg):
    if cfg.controller == "stop_and_turn":
        return StopAndTurnStrategy(cfg)
    if cfg.controller == "pid":
        return PIDController(cfg)
    if cfg.controller == "bangbang":
        return BangBangController(cfg)
    raise ValueError("Unknown controller: %s" % cfg.controller)


class LoopTimer:
    """Measures the real loop period, which the PID needs for dt."""

    def __init__(self):
        self.prev = None

    def tick(self):
        now = time.perf_counter()
        dt = 0.0 if self.prev is None else now - self.prev
        self.prev = now
        return dt
