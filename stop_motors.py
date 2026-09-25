#!/usr/bin/env python3
"""
stop_motors.py -- Emergency stop. Forces every motor pin low and releases them.

Run this any time the wheels are spinning and you want them to stop:

    python3 stop_motors.py

It is deliberately tiny and has no dependency on the rest of the project, so
it still works if something else has crashed. Keep a terminal open with this
command typed and ready while you are testing -- pressing Enter is faster than
unplugging a battery pack with the car moving.

It also runs automatically at boot if you install the service described in
README section 11.
"""

import sys

PINS = [17, 18, 22, 23, 12, 13]      # direction pins + the optional ENA/ENB

try:
    import RPi.GPIO as GPIO
except Exception as exc:
    sys.exit("RPi.GPIO not available (%s) -- nothing to stop." % type(exc).__name__)

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

for pin in PINS:
    try:
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.output(pin, GPIO.LOW)
    except Exception as exc:
        print("  pin %d: %s" % (pin, exc))

print("All motor pins driven LOW.")

# Leave the pins configured as low outputs rather than calling GPIO.cleanup().
# cleanup() returns them to inputs, which lets them float again -- and a
# floating input on the L298N is exactly what makes the motors twitch.
