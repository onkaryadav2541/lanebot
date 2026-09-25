# Autonomous Lane-Following Vehicle with PID Lateral Control



## 1. Read this before you run anything

**Your exposé says L293D, your build uses an L298N.** Both are H-bridges and
the code works with either, but you must be consistent in the written thesis.
The L298N is the better part (higher current, less voltage drop). The simplest
fix is one sentence in the hardware chapter: the design was migrated from the
L293D to the L298N because the yellow gearbox motors stall the L293D's 600 mA
per channel. Then keep L298N everywhere else.

**The ENA/ENB jumpers block proportional steering.** This is the one thing
that can quietly ruin the whole thesis. A PID controller produces a continuous
command. If the motors can only be fully on or fully off, that continuous
command is thrown away at the actuator and what actually drives the car is
still bang-bang control — you would be writing about a PID that the hardware
cannot express.

Three options, in order of preference:

| `pwm_mode` | Wiring change | Quality |
|---|---|---|
| `enable_pwm` | Pull off both black jumper caps, run **ENA → GPIO 12 (pin 32)** and **ENB → GPIO 13 (pin 33)** | Best. Two wires. **Recommended.** |
| `direction_pwm` | None — keep your current wiring | Works. PWM is applied to IN1/IN3 instead. Slightly coarser at low speed. |
| `bangbang` | None | Reproduces the research paper. Keep it for the comparison chapter. |

The default in `config.py` is `direction_pwm`, so the code runs on your car as
wired today. Move the two jumper wires when you can and switch to
`enable_pwm`.

**Two more things worth checking on the bench:**

* The paper says the vehicle has two DC motors; your build has four. Update
  the hardware chapter — four motors on two channels changes the current draw
  and the effective track width used in the kinematics.
* A 6 V NiMH pack sags under four stalled motors. If the Pi reboots when the
  motors start, that is a power problem, not a software one.

---

## 2. Install (on the Raspberry Pi)

```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy python3-picamera2 python3-rpi.gpio
pip3 install matplotlib          # only needed for the plots in benchmark/simulate
```

Copy the whole `lanebot/` folder to the Pi (e.g. `/home/pi/lanebot`), open it
in Thonny, and run `selftest.py` first.

On Bookworm, `picamera2` and `RPi.GPIO` come from apt, not pip. If
`RPi.GPIO` misbehaves on Bookworm, `sudo apt install python3-rpi-lgpio`
provides a drop-in replacement.

---

## 3. First run, in order

```bash
python3 selftest.py                    # 1. software check, no hardware needed
python3 test_hardware.py --camera      # 2. camera works, real capture fps
python3 test_hardware.py --vision      # 3. is the lane visible after threshold?
python3 test_hardware.py --ramp        # 4. wheels up! find motor_min_duty
python3 test_hardware.py --motors      # 5. wheels up! check directions
python3 main.py --no-motors            # 6. full pipeline, motors disabled
python3 main.py                        # 7. press 's' to arm, SPACE to e-stop
```

Steps 4 and 5 must be done with the car on a box so the wheels spin freely.

### Keys while `main.py` runs

| Key | Action | Key | Action |
|---|---|---|---|
| `s` | arm / disarm motors | `SPACE` | emergency stop |
| `a` | switch algorithm A ↔ B | `c` | switch PID ↔ bang-bang |
| `t` | cycle threshold mode | `d` | debug windows |
| `1`/`2` | Kp − / + | `3`/`4` | Ki − / + |
| `5`/`6` | Kd − / + | `[`/`]` | base speed − / + |
| `r` | reset PID state | `q` | quit and print the summary |

Every run writes `runs/<name>_<timestamp>/` containing `frames.csv` (one row
per frame), `summary.json` and `config.json`. Those three files are your raw
data for the results chapter — never delete them.

---

## 4. Files

| File | Role |
|---|---|
| `config.py` | every tunable parameter, saved with each run |
| `camera.py` | PiCamera2 / USB / video sources, threaded capture |
| `vision.py` | grayscale → blur → threshold → Canny → ROI, with per-stage timing |
| `detectors.py` | **Algorithm A** Canny+Hough, **Algorithm B** histogram |
| `controller.py` | PID (anti-windup, filtered derivative) + bang-bang + speed mixer |
| `motors.py` | L298N driver, three PWM strategies, mock driver off-Pi |
| `dashboard.py` | telemetry panel, PID term bars, digital twin |
| `metrics.py` | FPS, latency, deviation, oscillation, CSV + JSON output |
| `main.py` | the control loop |
| `benchmark.py` | offline A vs B comparison on identical footage |
| `simulate.py` | closed-loop SIL evaluation (H1, H2) |
| `tune_pid.py` | Ziegler-Nichols gain search + analysis of recorded runs |
| `test_hardware.py` | bring-up checks |
| `selftest.py` | 27 assertions over the whole stack |

---

## 5. How the code answers the exposé

| Exposé item | Where |
|---|---|
| Algorithm A: Canny + Hough | `detectors.CannyHoughDetector` |
| Algorithm B: histogram | `detectors.HistogramDetector` |
| PID with Kp, Ki, Kd | `controller.PIDController` |
| Ziegler-Nichols tuning | `tune_pid.py --mode zn` |
| FPS and end-to-end latency | `metrics.py`, measured capture → motor command |
| Navigation success | detection rate + in-lane fraction |
| Steering accuracy | mean / RMS lateral deviation, `summary.json` |
| Environmental robustness | threshold-mode sweep, `benchmark.py --thresholds` |
| Dynamic thresholding (H3) | `vision.Preprocessor`: fixed / Otsu / adaptive |
| Real-time optimisations | threaded capture, ROI-first processing, dashboard throttling |
| Digital twin + telemetry | `dashboard.py` |

### Testing the four hypotheses

**H1 — histogram gives smoother steering than Canny-Hough.**
Perception quality on identical footage:
```bash
python3 main.py --record --no-motors        # record a clip first
python3 benchmark.py --video runs/<...>/raw.avi
```
Closed-loop consequence, which is what "smoother steering" actually means:
```bash
python3 simulate.py --compare-algorithms
python3 simulate.py --compare-controllers   # PID vs bang-bang, the stronger result
```

**H2 — latency above 100 ms degrades PID performance.**
```bash
python3 simulate.py --latency-sweep --hz 30
```
Report the measured on-vehicle latency from `summary.json` next to the sweep.
Note in the text that the delay buffer is quantised to whole control periods.

**H3 — dynamic thresholding is essential under variable lighting.**
Record the same track under three lighting conditions, then:
```bash
python3 benchmark.py --video clip_bright.avi --thresholds --sources
python3 benchmark.py --video clip_shadow.avi --thresholds --sources
```

Read this part carefully, because it decides whether H3 is testable at all.
The histogram can be summed over the Canny edge image, over the binary
threshold image, or over both (`VisionConfig.hist_source`). **If it sums only
over Canny edges, the binary threshold has no influence on the result and a
thresholding sweep will show three identical rows** — not because dynamic
thresholding is useless, but because nothing in that pipeline uses the
threshold. `--sources` sweeps this dimension too, which is why the default is
`combined`.

In a shadowed clip the expected pattern is that `fixed/binary` loses detections
while `otsu/binary` and `adaptive/binary` hold up. The honest conclusion is
narrower than the hypothesis as written: dynamic thresholding matters for the
intensity-threshold path, and is largely irrelevant to the gradient-based Canny
path, because Canny responds to local contrast rather than absolute brightness.
Write that. A hypothesis that is confirmed only under stated conditions is a
stronger result than one waved through, and this is the kind of nuance that
distinguishes a good thesis in the defence.

**H4 — ≥ 15 FPS on a Pi 3B+.**
Read `fps_mean` and the stage breakdown from `summary.json`. If you fall short,
the two levers already in the code are `dashboard.update_every` (rendering is
usually the largest single cost) and `camera.threaded`. Report the numbers with
and without each — that is exactly the "optimisation strategies" question in
the exposé.

---

## 6. PID tuning workflow

1. **Get a starting point.**
   ```bash
   python3 tune_pid.py --mode zn
   ```
   This raises Kp with Ki = Kd = 0 until the simulated vehicle oscillates at
   constant amplitude, reads off the critical gain Ku and the period Tu, and
   prints the four classic Ziegler-Nichols rules. The defaults shipped in
   `config.py` (Kp 1.60, Ki 4.00, Kd 0.20) came from the "no overshoot" rule
   plus a sweep, and hold the simulated car within about 2 cm of the lane
   centre on a mixed track.

2. **Refine on the vehicle.** Start slow (`--speed 0.4`), arm the motors, and
   adjust with keys `1`–`6`. Classic symptoms:

   | Symptom | Fix |
   |---|---|
   | fast weaving around the centre | lower Kp ~30 %, or raise Kd |
   | drives consistently off-centre | raise Ki — but first check the camera is not tilted |
   | overshoots into corners then snaps back | raise Kd, lower `base_speed` |
   | sluggish, cuts corners wide | raise Kp |
   | oscillates only at speed | gains are speed dependent; retune at the target speed |

3. **Prove it with data.**
   ```bash
   python3 tune_pid.py --mode analyse --csv runs/<run>/frames.csv
   ```
   Reports bias, RMS error, dominant oscillation period and saturation time.

Report both the ZN values *and* the field-refined values in the thesis, with
the reason for each change. That contrast is the "PID Tuning Documentation"
deliverable in section 4 of the exposé.

---

## 7. Parameters you will most likely need to change

| Parameter | Where | When |
|---|---|---|
| `motor_min_duty` | `MotorConfig` | wheels hum but do not turn → raise it |
| `threshold_mode` | `VisionConfig` | shadows on the track → `adaptive` |
| `invert_binary` | `VisionConfig` | dark lane lines on light paper |
| `roi_top_ratio` | `VisionConfig` | camera mounted higher or lower |
| `expected_lane_width` | `VisionConfig` | different lane spacing on your track |
| `hist_min_peak` | `VisionConfig` | detection drops out → lower it |
| `base_speed` | `ControlConfig` | always start low and work up |
| `flip_vertical` | `CameraConfig` | camera mounted upside down |

---

## 8. Honest limitations to state in the thesis

Examiners reward these; hiding them is what gets punished.

* The closed-loop results from `simulate.py` come from a kinematic model, not
  the road. The perception and control code is the real code, but the vehicle
  dynamics — wheel slip, motor lag, battery sag — are idealised. Label those
  figures as software-in-the-loop and keep them separate from on-vehicle runs.
* The synthetic track generator is a clean, high-contrast image. It is useful
  for regression testing and for tuning, not for claims about robustness.
* Lateral deviation is measured in pixels of image error, not surveyed ground
  truth. Convert to cm using a measured pixels-per-cm scale and say how you
  measured it.
* Bang-bang and PID must be compared at the same speed, on the same track, in
  the same lighting, or the comparison means nothing.

---

## 9. Testing at home: what to print and how to check it works

### What to print

```bash
python3 make_track.py --all
```

Writes three PDFs into `printables/`:

* **`bench_sheets_a4.pdf`** — 11 sheets: centred lane, offset left, offset
  right, gentle and sharp curves in both directions, dashed markings, narrow
  and wide lane, and a single-boundary sheet. This is what you actually test
  with.
* **`calibration_a4.pdf`** — a 200 mm ruler, a 20 mm checkerboard and a lane of
  known width. Needed to convert pixels to centimetres.
* **`floor_tiles_a4.pdf`** — tiles for a driving track.

**Print at 100 %, never "fit to page".** Every sheet carries a 100 mm reference
bar — measure it with a ruler after printing. If it comes out at 96 mm your
printer scaled the page, and every distance in your results chapter is wrong by
that factor.

**Use matt paper, not glossy.** Glossy paper mirrors the ceiling lights straight
into the camera, and a specular highlight looks exactly like a lane marking to
any threshold-based detector. If you only have glossy paper, kill the ceiling
light and use a lamp off to one side.

### What lane width?

The default is 180 mm centre-to-centre with 15 mm lines, which fits A4 with room
left for the offset sheets. The rule that actually matters: **the lane should
fill roughly half to two thirds of the image width** at the distance the ROI
looks at. Too narrow and the two histogram peaks sit close together, so small
errors vanish into noise; too wide and one boundary leaves the frame on every
curve.

Don't guess it — measure it:

```bash
python3 calibrate.py --mode place
```

Live view with the ROI band drawn, the detected boundaries marked, and written
advice ("lane too narrow, move the camera closer", "only one boundary visible,
move back"). Adjust the camera bracket until it reads GOOD PLACEMENT, then copy
the measured lane width into `VisionConfig.expected_lane_width`.

For driving on the floor, **black masking or electrical tape on a light floor
beats printed tiles**: cheaper, it doesn't curl, it survives being driven over,
and you can lay a curve of any radius. Two parallel strips 20–25 cm apart. Use
the printed tiles only if you want a documented, photographable track.

### Is the car actually working? The check sequence

```bash
python3 calibrate.py --mode check
```

Thirty seconds, and it prints a pass/fail list: camera frame rate, detection
rate, end-to-end latency against the 80 ms threshold, CPU temperature, Pi
throttling and under-voltage flags, GPIO availability. Where a check fails it
tells you what to change. Run it before every test session and keep the output —
it's your evidence that the platform was healthy when the data was recorded.

Then work through the bench sheets:

1. Tape a sheet to the wall, or lay it flat, and aim the camera at it.
2. `python3 main.py --no-motors` and watch the dashboard.
3. **Slide the sheet slowly left and right.** The error number, the offset bar,
   the digital twin and the error graph must all follow the sheet. If they do,
   perception and control are working — that is the core validation, and it is
   exactly the software-in-the-loop argument your research paper already made.
4. Swap to the curve sheets and check the command switches to LEFT/RIGHT in the
   right direction. Wrong direction means `swap_sides` or a camera flip.
5. `python3 main.py --record --no-motors` on each sheet, so `benchmark.py` has
   footage to compare the two algorithms on.

Then convert pixels to centimetres, so the results aren't in arbitrary units:

```bash
python3 calibrate.py --mode scale --known-mm 200
python3 report.py --run runs/<your run> --px-per-cm 12.4
```

### Thermal behaviour

The exposé lists CPU throttling and thermal management as a technical
challenge, so measure it rather than asserting it:

```bash
python3 calibrate.py --mode thermal --minutes 15
```

Logs CPU temperature, ARM clock, throttling flags and frame rate, and produces
`thermal.png`. On a bare Pi 3B+ without a heatsink you will usually watch the
temperature climb past 80 °C and the frame rate fall with it. That is a real
measured result for the constraints chapter, and one examiners rarely see.

---

## 10. Figures for the thesis

The live dashboard is for driving. **Don't screenshot it for the document** — a
15 Hz OpenCV window looks amateur in print. Every run writes `frames.csv`; turn
that into proper figures:

```bash
# single run
python3 report.py --run runs/histogram_pid_20260904_141233

# comparison, which is what examiners actually want to see
python3 report.py --run runs/<pid run> --run runs/<bangbang run> \
                  --labels "PID" "Bang-bang" --name pid_vs_bangbang
```

Fourteen figures come out as both PNG (300 dpi, for Word) and PDF (vector, for
LaTeX), plus `metrics_table.md`, `metrics_table.tex` and `figure_captions.md`
with a written caption for each.

| # | Figure | What it proves |
|---|---|---|
| 1 | Lateral error over time | the basic tracking result |
| 2 | Error distribution with normal fit | bias and noise, separated |
| 3 | Cumulative distribution of \|error\| | p95 accuracy, stricter than a mean |
| 4 | PID term contributions | that P, I and D each do something — the tuning chapter |
| 5 | Steering command vs error | **the single best figure**: bang-bang gives three discrete levels, PID a continuous curve |
| 6 | Frame rate over time and histogram | H4 |
| 7 | Latency distribution with 80/100 ms lines | H2 |
| 8 | Per-stage timing breakdown and share | where the frame budget goes, and what to optimise |
| 9 | Detection rate and confidence | robustness |
| 10 | Motor duty cycles | that the controller actually reaches the actuator |
| 11 | Detected boundaries and lane centre | perception quality |
| 12 | Power spectrum of the error | oscillation frequency — the quantitative form of H1 |
| 13 | Phase portrait | stability argument, and it looks impressive |
| 14 | Latency over the run | reveals throttling and CPU contention |

**One trap.** Comparing PID against bang-bang on the *same recorded video* is
meaningless: the frames don't react to steering, so both controllers see an
identical error signal and figures 1, 2 and 3 come out as identical lines. From
recorded video only figure 5 and the steering-activity metric are valid. For a
real controller comparison either drive the car twice under the same
conditions, or use `simulate.py --compare-controllers` — and state in the text
which one produced each figure.

### What the live dashboard now shows

Algorithm, controller, error in px, motor command, an offset bar with the
deadzone marked, the digital twin with a motion trail, FPS, latency (green
under 80 ms, red over), detection confidence, left and right duty cycles, a
lane-histogram bar plot with the detected peaks marked, and four scrolling
charts: lateral error, PID terms, motor duties, latency and FPS. `--no-graphs`
turns the charts off if you need the frame rate back.

### Every reading the code records

`frames.csv` has one row per frame with: frame index, timestamp, algorithm,
controller, detected, inferred, confidence, lane centre, left and right
boundary positions, raw error, smoothed error, steer command, motor command,
P/I/D terms, left and right duty, end-to-end latency, loop time, FPS, and the
execution time of every pipeline stage. `summary.json` aggregates these into
frame counts, mean and minimum FPS, mean/p95/max latency, detection rate, mean
confidence, mean/RMS/max/std lateral error, zero-crossing counts, per-stage
timings, and explicit H2 and H4 pass flags.

---

---

## 11. Stopping the car running away on power-up

If the wheels spin the instant you connect the battery, the L298N input pins
are not being held low. Between the moment the Pi gets power and the moment
your script calls `GPIO.setup(..., initial=GPIO.LOW)`, those pins are inputs.
An input is high impedance, it floats, and a floating L298N input is read as
whatever noise happens to be on the wire. With the ENA/ENB jumpers fitted, a
floating IN pin is enough to drive a motor.

Apply as many of these as you can. The first is the one that matters.

### 1. A switch in the battery lead (do this one)

Put a slide switch or a barrel connector in the **positive** wire between the
6 V pack and the L298N +12V terminal. Logic stays powered from the Pi, so the
software keeps running and the dashboard keeps updating, but the motors cannot
move until you close the switch.

No software fix is as reliable as being able to physically cut motor power in
half a second, and every examiner who has watched a robot demo will recognise
it as the right engineering decision.

### 2. Drive the pins low from early boot

Raspberry Pi OS can configure GPIO state before userspace starts. Edit the boot
config:

```bash
sudo nano /boot/firmware/config.txt      # older images: /boot/config.txt
```

Add at the end:

```
gpio=17,18,22,23=op,dl
```

`op` means output, `dl` means driven low. Reboot. The pins are now held at 0 V
from very early in boot instead of floating, which closes most of the window
where the car can twitch.

### 3. Pull-down resistors (the proper hardware fix)

Fit a 10 kOhm resistor from each of IN1, IN2, IN3, IN4 to ground. They hold the
L298N inputs at 0 V whenever the Pi is not actively driving them, including
while the Pi is switched off entirely. This is what a production board would
do, and it costs four resistors.

On the breadboard: one leg of each resistor into the row carrying that IN
signal, the other leg into your ground rail at row 10.

### 4. Emergency stop from the keyboard

Two are already built in:

* `python3 stop_motors.py` forces every motor pin low. Keep a second terminal
  open with this typed and ready while testing.
* In `main.py`, the motors start **disarmed**. Nothing moves until you press
  `s`, and `SPACE` is an immediate emergency stop. The car also stops itself
  automatically if the lane is lost for more than 20 frames.

### 5. Optional: hold the pins low at every boot

```bash
sudo nano /etc/systemd/system/motors-safe.service
```

```ini
[Unit]
Description=Hold the motor pins low at boot
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/python3 /home/onkaryadav23042000/lanebot/stop_motors.py

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable motors-safe.service
```

### Habit that prevents most of it

Connect the battery **last**, after the Pi has booted and your script is
running and showing the dashboard. Disconnect it **first**, before you close
anything down. And keep the car on a box with the wheels free until you have
watched it steer correctly on the bench.
