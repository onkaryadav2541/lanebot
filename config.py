"""
config.py -- Central configuration for the autonomous lane-following vehicle.

Every tunable parameter of the system lives here so that experiments are
reproducible: a single Config object fully describes a test run and can be
dumped to / loaded from JSON alongside the results CSV.

Author: Onkar Yadav (Matr. 100002351), SRH Hochschule Heidelberg
"""

from dataclasses import dataclass, asdict, field
import json


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------
@dataclass
class CameraConfig:
    source: str = "picamera"      # "picamera" | "usb" | "video"
    video_path: str = ""          # used when source == "video"
    usb_index: int = 0            # used when source == "usb"
    width: int = 320
    height: int = 240
    fps_target: int = 30
    threaded: bool = True         # decouple capture from processing (latency opt.)
    flip_horizontal: bool = False
    flip_vertical: bool = True    # measured: camera is mounted upside down


# --------------------------------------------------------------------------
# Vision pipeline
# --------------------------------------------------------------------------
@dataclass
class VisionConfig:
    # --- preprocessing -----------------------------------------------------
    blur_kernel: int = 7                 # must be odd (7x7 as in the paper)
    threshold_mode: str = "otsu"         # "fixed" | "otsu" | "adaptive"  (H3)
    fixed_threshold: int = 140
    adaptive_block: int = 31             # must be odd
    adaptive_c: int = -5
    # TRUE for dark markings on a light surface (black tape on white paper),
    # FALSE for light markings on a dark surface (white tape, or real road
    # paint on asphalt). Getting this wrong makes the detector lock onto the
    # background instead of the lane: it reports a high confidence and a
    # constant, badly wrong error.
    invert_binary: bool = True
    canny_low: int = 50
    canny_high: int = 150
    use_clahe: bool = False              # local contrast equalisation (H3)
    clahe_clip: float = 2.0

    # --- region of interest ------------------------------------------------
    # A taller ROI sees further ahead, so a curve is noticed before the
    # vehicle has already entered it. 0.60 uses the bottom 40% of the frame.
    roi_top_ratio: float = 0.40
    roi_bottom_ratio: float = 1.0

    # --- histogram detector (Algorithm B) ----------------------------------
    # Which mask the histogram is summed over. IMPORTANT for hypothesis H3:
    # with "edges" the histogram sees only the Canny output, so the binary
    # threshold has no effect on the result at all and a thresholding sweep
    # would show no difference. "combined" (binary OR edges) makes both
    # contribute, which is what lets H3 be tested. "edges" reproduces the
    # Canny-Histogram pipeline of the research paper.
    hist_source: str = "combined"           # "edges" | "binary" | "combined"
    hist_margin: int = 10                # ignore this many px at each border
    # Lower threshold: in a curve one boundary is often faint or partly out
    # of frame, and 500 was rejecting it. The detector can work from a single
    # boundary by inferring the other from expected_lane_width.
    hist_min_peak: int = 200
    hist_smooth: int = 5                 # moving-average window on histogram

    # --- Canny + Hough detector (Algorithm A) ------------------------------
    hough_threshold: int = 20
    hough_min_line_len: int = 15
    hough_max_line_gap: int = 20
    min_abs_slope: float = 0.30          # reject near-horizontal lines

    # --- shared ------------------------------------------------------------
    expected_lane_width: int = 235       # px, measured with calibrate.py --mode place
    lane_width_alpha: float = 0.05       # online adaptation of the above

    # Plausibility gate on the detected lane width. A detection whose width
    # differs from expected_lane_width by more than this fraction is rejected
    # as a false positive. Without it the detector can lock onto the edges of
    # the paper or the image border -- which produces a confident, stable and
    # completely wrong lane, and therefore results that look excellent while
    # measuring nothing. 0.45 accepts roughly 66 to 174 px for a 120 px lane,
    # which covers genuine perspective change on a curve.
    lane_width_tolerance: float = 0.60


# --------------------------------------------------------------------------
# Control
# --------------------------------------------------------------------------
@dataclass
class ControlConfig:
    # "pid"           continuous differential steering (the thesis contribution)
    # "bangbang"      LEFT / STRAIGHT / RIGHT (the research-paper baseline)
    # "stop_and_turn" drive straight, stop and rotate on the spot at a bend,
    #                 then drive straight again. Use this when the drivetrain
    #                 cannot produce enough differential to arc while moving.
    controller: str = "pid"

    # PID gains operate on the NORMALISED error e = error_px / (width/2),
    # so e is in [-1, 1] and the gains are resolution independent.
    # Starting values obtained with tune_pid.py (Ziegler-Nichols ultimate
    # gain search, Ku = 7.7, Tu = 0.43 s, "no overshoot" rule) and then
    # refined by a sweep over the closed-loop simulator. Re-run
    #     python3 tune_pid.py --mode zn
    # after any change to speed, camera height or track, because the gains
    # are only valid for the plant they were tuned on.
    kp: float = 4.00
    ki: float = 1.50
    kd: float = 0.20

    deadzone_px: float = 0.0      # much smaller than the paper's 30 px: a PID
                                  # does not need a wide dead band to stay
                                  # stable, and a wide one costs accuracy
    integral_limit: float = 0.40  # anti-windup clamp on the I term
    output_limit: float = 1.00    # steer command saturation
    d_filter_alpha: float = 0.30  # low-pass on the derivative term

    # bang-bang fallback (reproduces the research-paper behaviour)
    bangbang_deadzone_px: float = 30.0

    # --- stop-and-turn strategy -------------------------------------------
    pivot_enter_px: float = 22.0   # error above this -> stop and rotate
    pivot_exit_px: float = 7.0     # error below this -> resume driving
    pivot_pulse_s: float = 0.35    # length of one rotation burst
    pivot_pause_s: float = 0.30    # look between bursts, lets vision catch up
    pivot_speed: float = 1.00      # duty during a pivot (needs full torque)
    drive_speed: float = 0.95      # duty when driving straight

    # --- stop marker ------------------------------------------------------
    # A band of tape laid right across the lane. Two lane markings fill only a
    # small fraction of the region of interest; a band across the whole lane
    # fills most of it, so a simple coverage test separates the two reliably.
    # On seeing one the vehicle halts, rotates for a fixed time, drives clear
    # of the marker and resumes normal lane following. A fixed rotation is
    # used rather than a visual search because it is deterministic: it cannot
    # be misled by carpet, shadows or the edge of the paper.
    marker_enabled: bool = False
    # The marker is the lane itself, widened. Where a turn begins, the two
    # lane markings are laid thicker (for example 5 cm instead of 2 cm). The
    # track still reads as an ordinary lane, and no foreign object has to be
    # placed in the driving path. The detector measures the thickness of each
    # marking and triggers when it exceeds this many pixels. Read the live
    # value from the dashboard ("w 12" next to the confidence) on the thin and
    # on the thick section, then set this threshold between the two.
    marker_line_width_px: float = 22.0
    marker_frames: int = 3          # consecutive frames before it triggers
    marker_turn_s: float = 1.20     # rotation time; tune to your corner angle
    marker_turn_direction: str = "right"   # "right" | "left"
    marker_clear_s: float = 0.90    # drive forward to get off the marker
    marker_cooldown_s: float = 2.50 # ignore markers for this long afterwards

    # --- lane search, used when the lane leaves the field of view ---------
    search_pulse_s: float = 0.40        # longer bursts: a search must sweep
    search_pause_s: float = 0.25        # pause so the camera can catch up
    search_reverse_after_s: float = 3.0 # sweep back the other way after this
    reacquire_frames: int = 3           # consecutive good frames to resume
    reacquire_min_conf: float = 0.25    # below this a detection is a glimpse

    # --- speed mixing ------------------------------------------------------
    base_speed: float = 0.45      # 0..1 nominal forward duty. High because the
                                  # 6 V pack minus the L298N drop leaves the
                                  # motors barely enough torque to move.
    max_speed: float = 1.00
    min_speed: float = -0.35       # the inner wheel must keep driving through a
                                  # turn. At 0.00 it stops dead and the car
                                  # pivots on the spot instead of steering
                                  # round the curve.
    steer_gain: float = 0.90      # how much of `steer` is mixed into wheels
    turn_slowdown: float = 0.60   # reduce base speed proportional to |steer|

    # --- safety ------------------------------------------------------------
    # When a curve is entered, the outer boundary leaves the field of view
    # before the vehicle has finished turning. Holding the last steering
    # command through that gap lets the car continue into the bend until the
    # lane reappears, instead of stopping in the middle of the corner.
    lost_lane_hold_frames: int = 45   # keep last command this many frames ...
    lost_lane_stop_frames: int = 90   # ... then stop the vehicle
    smoothing_alpha: float = 0.30     # IIR for dashboard display only


# --------------------------------------------------------------------------
# Motors (L298N dual H-bridge)
# --------------------------------------------------------------------------
@dataclass
class MotorConfig:
    enabled: bool = True

    # Direction pins (BCM numbering) -- matches your wiring notes
    left_in1: int = 17            # physical pin 11 -> IN1
    left_in2: int = 18            # physical pin 12 -> IN2
    right_in1: int = 22           # physical pin 15 -> IN3
    right_in2: int = 23           # physical pin 16 -> IN4

    # Speed control strategy:
    #   "enable_pwm"    -> remove the ENA/ENB jumper caps and wire them to
    #                      the GPIOs below. Best quality, true analogue speed.
    #   "direction_pwm" -> keep the ENA/ENB jumpers on and PWM the IN pins.
    #                      Works out of the box with your current wiring.
    #   "bangbang"      -> pure digital ON/OFF (research-paper behaviour).
    pwm_mode: str = "enable_pwm"
    ena_pin: int = 12             # only used in "enable_pwm" mode
    enb_pin: int = 13             # only used in "enable_pwm" mode
    pwm_frequency: int = 1000     # Hz

    # Gearbox motors do not turn below a certain duty cycle. Anything the
    # controller asks for is remapped into [motor_min_duty, 100].
    motor_min_duty: float = 50.0
    deadband: float = 0.03        # |speed| below this -> motor off

    invert_left: bool = False     # flip if a side spins backwards
    invert_right: bool = True
    swap_sides: bool = False      # flip if left/right are wired the other way


# --------------------------------------------------------------------------
# Dashboard / logging
# --------------------------------------------------------------------------
@dataclass
class DashboardConfig:
    enabled: bool = True
    headless: bool = False        # True when running over SSH without X
    panel_width: int = 160
    display_scale: float = 1.0    # 2.0 is very slow over VNC
    show_debug_windows: bool = False   # binary / edges / histogram windows
    update_every: int = 3         # render only every Nth frame (perf. option)
    show_graphs: bool = True      # live scrolling charts under the camera view
    show_histogram_strip: bool = True
    graph_height: int = 190
    graph_history: int = 220      # samples kept in each live chart


@dataclass
class LoggingConfig:
    enabled: bool = True
    directory: str = "runs"
    record_video: bool = False
    record_fps: int = 15


# --------------------------------------------------------------------------
# Root config
# --------------------------------------------------------------------------
@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    motors: MotorConfig = field(default_factory=MotorConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    algorithm: str = "histogram"   # "histogram" (B) | "canny_hough" (A)
    run_name: str = "run"

    # ---------------------------------------------------------------- utils
    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            data = json.load(fh)
        return cls(
            camera=CameraConfig(**data.get("camera", {})),
            vision=VisionConfig(**data.get("vision", {})),
            control=ControlConfig(**data.get("control", {})),
            motors=MotorConfig(**data.get("motors", {})),
            dashboard=DashboardConfig(**data.get("dashboard", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            algorithm=data.get("algorithm", "histogram"),
            run_name=data.get("run_name", "run"),
        )
