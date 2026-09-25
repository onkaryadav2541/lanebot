"""
motors.py -- L298N dual H-bridge driver.

Wiring assumed (matches your build notes, BCM numbering):

    GPIO 17 (pin 11) -> IN1   left side forward
    GPIO 18 (pin 12) -> IN2   left side backward
    GPIO 22 (pin 15) -> IN3   right side forward
    GPIO 23 (pin 16) -> IN4   right side backward
    Pi pin 2 (5V)    -> L298N +5V logic
    Pi pin 6 (GND)   -> common ground with the 6 V battery pack

IMPORTANT -- proportional steering needs a proportional actuator.

A PID controller outputs a continuous command. If the motors can only be
fully ON or fully OFF, the continuous command is destroyed at the actuator
and what actually runs on the vehicle is still bang-bang control. Three
strategies are therefore provided:

  pwm_mode = "enable_pwm"     Remove the black ENA and ENB jumper caps and
                              wire ENA -> GPIO 12, ENB -> GPIO 13. The IN
                              pins then only set direction and the enable
                              pins carry the PWM. This is the textbook wiring
                              and gives the cleanest speed control. RECOMMENDED
                              for the thesis, it is a two-wire change.

  pwm_mode = "direction_pwm"  Keep the ENA/ENB jumpers where they are and PWM
                              the direction pin instead (IN1 = PWM, IN2 = 0
                              for forward). Works with your current wiring
                              with no rewiring at all, at the cost of slightly
                              coarser low-speed behaviour.

  pwm_mode = "bangbang"       Pure digital ON/OFF, i.e. the behaviour of the
                              research paper. Kept so the thesis can show the
                              before/after comparison on identical hardware.

Yellow gearbox motors do not start turning below roughly 35-45 % duty. Any
non-zero request is therefore remapped into [motor_min_duty, 100] so that a
small steering correction produces actual movement instead of a stalled motor
that only hums and drains the pack.
"""

import time

import numpy as np

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except Exception:                     # not on a Pi -> development machine
    GPIO = None
    GPIO_AVAILABLE = False


class MockMotorDriver:
    """Stand-in used off-Pi and when motors are disabled. Records commands."""

    available = False

    def __init__(self, cfg):
        self.cfg = cfg
        self.left_duty = 0.0
        self.right_duty = 0.0
        self._kick_until = 0.0
        self._was_moving = (False, False)

    # ------------------------------------------------------------------
    def _kickstart(self, left, right):
        """
        Applies a short high-duty burst when a stationary channel starts.

        The burst scales BOTH channels by the same factor, so the ratio
        between them is preserved. Kicking each channel independently to full
        duty would erase the very speed difference the controller asked for,
        and the vehicle would drive straight through every correction.
        """
        c = self.cfg
        if not getattr(c, "kickstart_enabled", False):
            self._was_moving = (left > 0, right > 0)
            return left, right

        now = time.perf_counter()
        moving = (left > 0, right > 0)
        starting = any(m and not w for m, w in zip(moving, self._was_moving))
        self._was_moving = moving

        peak = max(left, right)
        if starting and 0 < peak < c.kickstart_below:
            self._kick_until = now + c.kickstart_ms / 1000.0
        if peak <= 0:
            self._kick_until = 0.0

        if now < self._kick_until and peak > 0:
            gain = 100.0 / peak
            return min(100.0, left * gain), min(100.0, right * gain)
        return left, right

    def drive(self, left_speed, right_speed):
        self.left_duty, self.right_duty = self._kickstart(
            self._duty(left_speed), self._duty(right_speed))

    def _duty(self, speed):
        s = abs(float(np.clip(speed, -1.0, 1.0)))
        if s < self.cfg.deadband:
            return 0.0
        return self.cfg.motor_min_duty + s * (100.0 - self.cfg.motor_min_duty)

    def stop(self):
        self.left_duty = self.right_duty = 0.0
        self._kick_until = 0.0
        self._was_moving = (False, False)

    def cleanup(self):
        pass


class L298NDriver(MockMotorDriver):
    available = True

    def __init__(self, cfg):
        super().__init__(cfg)
        if not GPIO_AVAILABLE:
            raise RuntimeError("RPi.GPIO is not available on this machine")

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        li1, li2 = (cfg.left_in1, cfg.left_in2)
        ri1, ri2 = (cfg.right_in1, cfg.right_in2)
        if cfg.swap_sides:
            li1, li2, ri1, ri2 = ri1, ri2, li1, li2
        if cfg.invert_left:
            li1, li2 = li2, li1
        if cfg.invert_right:
            ri1, ri2 = ri2, ri1
        self.pins = dict(li1=li1, li2=li2, ri1=ri1, ri2=ri2)

        for p in self.pins.values():
            GPIO.setup(p, GPIO.OUT, initial=GPIO.LOW)

        self.mode = cfg.pwm_mode
        self._pwm = {}

        if self.mode == "enable_pwm":
            GPIO.setup(cfg.ena_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(cfg.enb_pin, GPIO.OUT, initial=GPIO.LOW)
            self._pwm["left"] = GPIO.PWM(cfg.ena_pin, cfg.pwm_frequency)
            self._pwm["right"] = GPIO.PWM(cfg.enb_pin, cfg.pwm_frequency)
            self._pwm["left"].start(0)
            self._pwm["right"].start(0)
        elif self.mode == "direction_pwm":
            # PWM lives on the forward direction pins
            self._pwm["li1"] = GPIO.PWM(li1, cfg.pwm_frequency)
            self._pwm["ri1"] = GPIO.PWM(ri1, cfg.pwm_frequency)
            self._pwm["li2"] = GPIO.PWM(li2, cfg.pwm_frequency)
            self._pwm["ri2"] = GPIO.PWM(ri2, cfg.pwm_frequency)
            for p in self._pwm.values():
                p.start(0)
        elif self.mode != "bangbang":
            raise ValueError("Unknown pwm_mode: %s" % self.mode)

    # ------------------------------------------------------------------
    def drive(self, left_speed, right_speed):
        self.left_duty, self.right_duty = self._kickstart(
            self._duty(left_speed), self._duty(right_speed))
        self._apply("left", left_speed, self.left_duty)
        self._apply("right", right_speed, self.right_duty)

    def _apply(self, side, speed, duty):
        forward = speed >= 0
        if side == "left":
            fwd_pin, rev_pin = self.pins["li1"], self.pins["li2"]
            fwd_key, rev_key = "li1", "li2"
        else:
            fwd_pin, rev_pin = self.pins["ri1"], self.pins["ri2"]
            fwd_key, rev_key = "ri1", "ri2"

        if self.mode == "enable_pwm":
            GPIO.output(fwd_pin, GPIO.HIGH if (duty > 0 and forward) else GPIO.LOW)
            GPIO.output(rev_pin, GPIO.HIGH if (duty > 0 and not forward) else GPIO.LOW)
            self._pwm[side].ChangeDutyCycle(float(np.clip(duty, 0, 100)))

        elif self.mode == "direction_pwm":
            active_key = fwd_key if forward else rev_key
            idle_key = rev_key if forward else fwd_key
            self._pwm[idle_key].ChangeDutyCycle(0.0)
            self._pwm[active_key].ChangeDutyCycle(float(np.clip(duty, 0, 100)))

        else:  # bangbang
            on = duty > 0
            GPIO.output(fwd_pin, GPIO.HIGH if (on and forward) else GPIO.LOW)
            GPIO.output(rev_pin, GPIO.HIGH if (on and not forward) else GPIO.LOW)

    # ------------------------------------------------------------------
    def stop(self):
        self.left_duty = self.right_duty = 0.0
        if self.mode == "enable_pwm":
            for p in self._pwm.values():
                p.ChangeDutyCycle(0.0)
        elif self.mode == "direction_pwm":
            for p in self._pwm.values():
                p.ChangeDutyCycle(0.0)
        for p in self.pins.values():
            GPIO.output(p, GPIO.LOW)

    def cleanup(self):
        try:
            self.stop()
            for p in self._pwm.values():
                p.stop()
        finally:
            GPIO.cleanup()


# ----------------------------------------------------------------------
def create_driver(cfg, force_mock=False):
    """Returns a real driver on the Pi, a mock elsewhere. Never raises."""
    if force_mock or not cfg.enabled or not GPIO_AVAILABLE:
        return MockMotorDriver(cfg)
    try:
        return L298NDriver(cfg)
    except Exception as exc:            # busy pins, permissions, ...
        print("[motors] falling back to mock driver: %s" % exc)
        return MockMotorDriver(cfg)
