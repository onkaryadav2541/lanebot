#!/usr/bin/env python3
"""
drive.py -- Manual remote control. Drive the car with the keyboard.

No camera, no lane detection, no PID. Just you and the motors. Use it to
check traction on a surface, confirm the motors turn the right way, find the
lowest duty the car will actually move at, and get a feel for how fast it
runs before you let the controller take over.

    python3 drive.py                 # window mode (needs a display)
    python3 drive.py --terminal      # terminal mode, works over SSH
    python3 drive.py --speed 0.9     # start at a different speed

Controls

    w   forward           a   turn left        SPACE  stop
    s   backward          d   turn right       q      quit
    x   spin left         c   spin right
    + / -   speed up / down (also = and _)
    1..9    set speed directly (1 = 10 %, 9 = 90 %, 0 = 100 %)

In window mode the car keeps moving while you hold a key and stops shortly
after you let go. In terminal mode each press moves the car for a moment;
hold the key down and the auto-repeat keeps it going.

SAFETY
  * First run with the wheels off the ground.
  * The car stops by itself if no key arrives for a moment, so a lost
    connection or a crashed terminal will not leave it driving.
  * SPACE is an immediate stop, and the motors are always released on exit.
"""

import argparse
import os
import sys
import time

from config import Config
from motors import create_driver

# key -> (left, right) as fractions of the current speed
MOVES = {
    "w": (1.0, 1.0),      # forward
    "s": (-1.0, -1.0),    # backward
    "a": (0.3, 1.0),      # arc left  (right wheel faster)
    "d": (1.0, 0.3),      # arc right (left wheel faster)
    "x": (-1.0, 1.0),     # spin left on the spot
    "c": (1.0, -1.0),     # spin right on the spot
}

NAMES = {
    "w": "forward", "s": "backward", "a": "left", "d": "right",
    "x": "spin left", "c": "spin right",
}

HOLD_TIMEOUT = 0.35       # stop this long after the last key


class Teleop:
    def __init__(self, cfg, speed):
        self.cfg = cfg
        self.speed = max(0.0, min(1.0, speed))
        self.motors = create_driver(cfg.motors)
        self.last_key_time = 0.0
        self.current = "stopped"
        if not self.motors.available:
            print("!! MOCK driver -- no GPIO. Nothing will actually move.\n")

    # ------------------------------------------------------------------
    def handle(self, ch):
        """Returns False when the user asks to quit."""
        if ch in ("q", "\x1b"):
            return False

        if ch == " ":
            self.stop("stopped")
            return True

        if ch in "0123456789":
            self.speed = 1.0 if ch == "0" else int(ch) / 10.0
            self.report()
            return True

        if ch in "+=":
            self.speed = min(1.0, self.speed + 0.05)
            self.report()
            return True

        if ch in "-_":
            self.speed = max(0.0, self.speed - 0.05)
            self.report()
            return True

        if ch in MOVES:
            left, right = MOVES[ch]
            self.motors.drive(left * self.speed, right * self.speed)
            self.last_key_time = time.time()
            if self.current != NAMES[ch]:
                self.current = NAMES[ch]
                self.report()
        return True

    def tick(self):
        """Stops the car if no key has arrived recently."""
        if (self.current != "stopped"
                and time.time() - self.last_key_time > HOLD_TIMEOUT):
            self.stop("stopped")

    def stop(self, label):
        self.motors.stop()
        if self.current != label:
            self.current = label
            self.report()

    def report(self):
        print("\r  %-12s speed %3.0f %%   L %3.0f %%  R %3.0f %%        "
              % (self.current, self.speed * 100,
                 self.motors.left_duty, self.motors.right_duty), end="")
        sys.stdout.flush()

    def shutdown(self):
        try:
            self.motors.stop()
            self.motors.cleanup()
        finally:
            print("\n\nMotors released.\n")


# ----------------------------------------------------------------------
def run_window(teleop):
    """Window mode: OpenCV grabs the keys. Needs a display."""
    import cv2
    import numpy as np

    cv2.namedWindow("drive")
    while True:
        panel = np.full((260, 420, 3), (32, 30, 28), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = [
            ("w  forward       s  backward", (215, 212, 208)),
            ("a  left          d  right", (215, 212, 208)),
            ("x  spin left     c  spin right", (215, 212, 208)),
            ("SPACE stop       q  quit", (215, 212, 208)),
            ("+ / -  speed     1..9 set speed", (150, 148, 145)),
        ]
        y = 28
        for text, colour in lines:
            cv2.putText(panel, text, (16, y), font, 0.46, colour, 1, cv2.LINE_AA)
            y += 24
        cv2.putText(panel, "%s" % teleop.current.upper(), (16, y + 22),
                    font, 0.8,
                    (120, 230, 140) if teleop.current != "stopped" else (90, 90, 200),
                    2, cv2.LINE_AA)
        cv2.putText(panel, "speed %3.0f %%   L %3.0f   R %3.0f"
                    % (teleop.speed * 100, teleop.motors.left_duty,
                       teleop.motors.right_duty),
                    (16, y + 52), font, 0.5, (80, 220, 240), 1, cv2.LINE_AA)
        cv2.imshow("drive", panel)

        key = cv2.waitKey(30) & 0xFF
        if key != 255:
            ch = chr(key) if 32 <= key < 127 else ("\x1b" if key == 27 else "")
            if ch and not teleop.handle(ch.lower()):
                break
        teleop.tick()
    cv2.destroyAllWindows()


def run_terminal(teleop):
    """Terminal mode: raw stdin. Works over SSH with no display."""
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if ready:
                ch = sys.stdin.read(1)
                if not teleop.handle(ch.lower()):
                    break
            teleop.tick()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Manual keyboard driving")
    ap.add_argument("--config")
    ap.add_argument("--speed", type=float, default=0.9,
                    help="starting speed, 0..1")
    ap.add_argument("--terminal", action="store_true",
                    help="force terminal mode (no window)")
    ap.add_argument("--min-duty", type=float,
                    help="override MotorConfig.motor_min_duty for this run")
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    if args.min_duty is not None:
        cfg.motors.motor_min_duty = args.min_duty

    print(__doc__.split("Controls")[1].split("SAFETY")[0])
    print("  motor_min_duty = %.0f %%   pwm mode = %s\n"
          % (cfg.motors.motor_min_duty, cfg.motors.pwm_mode))

    teleop = Teleop(cfg, args.speed)
    use_terminal = args.terminal or not os.environ.get("DISPLAY")
    try:
        if use_terminal:
            print("  terminal mode -- click this window and press keys\n")
            run_terminal(teleop)
        else:
            print("  window mode -- click the 'drive' window and press keys\n")
            run_window(teleop)
    except KeyboardInterrupt:
        pass
    finally:
        teleop.shutdown()


if __name__ == "__main__":
    main()
