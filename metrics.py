"""
metrics.py -- Measurement and logging.

Records exactly the four evaluation metrics listed in the exposé, section 3:

    Real-time performance   -> fps, end-to-end latency (capture -> motor command)
    Navigation success      -> fraction of frames with a valid lane detection
    Steering accuracy       -> mean / RMS absolute lateral deviation in pixels
    Environmental robustness-> detection rate and confidence per run, comparable
                               across lighting conditions

and two extra quantities that the hypotheses need:

    oscillation amplitude and zero-crossing rate  -> H1 (smoother steering)
    per-stage timing breakdown                    -> H2 / H4 (latency budget)

Every run writes:
    runs/<name>_<timestamp>/frames.csv    one row per frame
    runs/<name>_<timestamp>/summary.json  aggregated numbers for the thesis
    runs/<name>_<timestamp>/config.json   full configuration of the run
"""

import csv
import json
import os
import time
from collections import deque

import numpy as np


class MetricsRecorder:
    FIELDS = [
        "frame", "t_rel", "algorithm", "controller",
        "armed", "detected", "inferred", "confidence",
        "lane_center", "left_x", "right_x",
        "error_px", "smoothed_error_px", "steer", "command",
        "p_term", "i_term", "d_term",
        "left_duty", "right_duty",
        "latency_ms", "loop_ms", "fps",
        "t_grayscale", "t_blur", "t_threshold", "t_canny", "t_roi",
        "t_detect", "t_control", "t_render",
    ]

    def __init__(self, cfg, config_obj=None, window=60):
        self.cfg = cfg
        self.dir = None
        self._fh = None
        self._writer = None
        self.rows = []
        self.t0 = time.perf_counter()
        self.frame = 0

        self.fps_window = deque(maxlen=window)
        self.lat_window = deque(maxlen=window)
        self.err_window = deque(maxlen=window)

        self.latencies = []
        self.loop_times = []
        self.errors = []
        self.detections = []
        self.confidences = []
        self.armed_flags = []
        self.errors_armed = []
        self.detections_armed = []
        self.confidences_armed = []
        self.stage_totals = {}

        if cfg.enabled:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            name = getattr(config_obj, "run_name", "run") if config_obj else "run"
            self.dir = os.path.join(cfg.directory, "%s_%s" % (name, stamp))
            os.makedirs(self.dir, exist_ok=True)
            self._fh = open(os.path.join(self.dir, "frames.csv"), "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
            self._writer.writeheader()
            if config_obj is not None:
                config_obj.save(os.path.join(self.dir, "config.json"))

    # ------------------------------------------------------------------
    def add(self, record):
        self.frame += 1
        record.setdefault("frame", self.frame)
        record.setdefault("t_rel", time.perf_counter() - self.t0)

        lat = record.get("latency_ms", 0.0)
        loop = record.get("loop_ms", 0.0)
        self.latencies.append(lat)
        self.loop_times.append(loop)
        self.lat_window.append(lat)
        if loop > 0:
            self.fps_window.append(1000.0 / loop)
        self.detections.append(1 if record.get("detected") else 0)
        self.confidences.append(record.get("confidence", 0.0))
        armed = bool(record.get("armed"))
        self.armed_flags.append(1 if armed else 0)
        if armed:
            self.detections_armed.append(1 if record.get("detected") else 0)
            self.confidences_armed.append(record.get("confidence", 0.0))
            if record.get("detected"):
                self.errors_armed.append(record.get("error_px", 0.0))
        if record.get("detected"):
            self.errors.append(record.get("error_px", 0.0))
            self.err_window.append(record.get("error_px", 0.0))

        for key in ("t_grayscale", "t_blur", "t_threshold", "t_canny",
                    "t_roi", "t_detect", "t_control", "t_render"):
            if key in record:
                self.stage_totals.setdefault(key, []).append(record[key])

        if self._writer is not None:
            row = {k: record.get(k, "") for k in self.FIELDS}
            for k, v in row.items():
                if isinstance(v, float):
                    row[k] = round(v, 4)
            self._writer.writerow(row)

    # ------------------------------------------------------------------
    @property
    def fps(self):
        return float(np.mean(self.fps_window)) if self.fps_window else 0.0

    @property
    def latency(self):
        return float(np.mean(self.lat_window)) if self.lat_window else 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def _zero_crossings(seq):
        if len(seq) < 2:
            return 0
        s = np.sign(np.asarray(seq, dtype=np.float64))
        s[s == 0] = 1
        return int(np.sum(s[1:] != s[:-1]))

    def summary(self):
        # Statistics are computed over the frames in which the vehicle was
        # actually driving. Frames logged while the motors were disarmed
        # describe a stationary vehicle and would otherwise dominate the
        # averages, making a failed run look like an excellent one.
        armed = np.asarray(self.armed_flags, dtype=bool)
        n_armed = int(armed.sum())

        # Use the driven frames when there are any. Tracking accuracy measured
        # while the vehicle is stationary is not tracking accuracy at all.
        driven = n_armed > 0
        err = np.asarray(self.errors_armed if driven else self.errors,
                         dtype=np.float64)
        detections = self.detections_armed if driven else self.detections
        confidences = self.confidences_armed if driven else self.confidences
        lat = np.asarray(self.latencies, dtype=np.float64)
        loop = np.asarray(self.loop_times, dtype=np.float64)
        loop_nz = loop[loop > 0]

        duration = time.perf_counter() - self.t0
        # Throughput is the honest headline number: frames actually completed
        # divided by wall-clock time. The arithmetic mean of the instantaneous
        # rates is NOT the same thing and is badly inflated whenever some
        # frames are cheap (for example when the dashboard is rendered only
        # every Nth frame). Report throughput in the thesis.
        throughput = (self.frame / duration) if duration > 0 else 0.0
        out = {
            "frames": int(self.frame),
            "duration_s": round(duration, 2),
            "fps_throughput": round(float(throughput), 2),
            "fps_instantaneous_mean": round(float(np.mean(1000.0 / loop_nz)), 2)
            if loop_nz.size else 0.0,
            "fps_median": round(float(np.median(1000.0 / loop_nz)), 2)
            if loop_nz.size else 0.0,
            "fps_min": round(float(np.min(1000.0 / loop_nz)), 2) if loop_nz.size else 0.0,
            "latency_mean_ms": round(float(np.mean(lat)), 2) if lat.size else 0.0,
            "latency_p95_ms": round(float(np.percentile(lat, 95)), 2) if lat.size else 0.0,
            "latency_max_ms": round(float(np.max(lat)), 2) if lat.size else 0.0,
            "detection_rate": round(float(np.mean(detections)), 4) if detections else 0.0,
            "confidence_mean": round(float(np.mean(confidences)), 4) if confidences else 0.0,
            "mean_abs_error_px": round(float(np.mean(np.abs(err))), 2) if err.size else 0.0,
            "rms_error_px": round(float(np.sqrt(np.mean(err ** 2))), 2) if err.size else 0.0,
            "max_abs_error_px": round(float(np.max(np.abs(err))), 2) if err.size else 0.0,
            "error_std_px": round(float(np.std(err)), 2) if err.size else 0.0,
            "oscillation_zero_crossings": self._zero_crossings(
                self.errors_armed if driven else self.errors),
            "zero_crossings_per_s": round(
                self._zero_crossings(self.errors_armed if driven else self.errors)
                / duration, 3)
            if duration > 0 else 0.0,
        }
        # hypothesis checks
        out["frames_armed"] = n_armed
        out["frames_disarmed"] = int(armed.size - n_armed)
        if armed.size and n_armed == 0:
            out["WARNING"] = ("no frames were recorded with the motors armed: "
                              "this run describes a stationary vehicle")
        elif armed.size and n_armed < 0.60 * armed.size:
            out["WARNING"] = ("only %d of %d logged frames were driven (%.0f %%); "
                              "accuracy figures are computed over the driven "
                              "frames only"
                              % (n_armed, armed.size,
                                 100.0 * n_armed / armed.size))
        out["H2_latency_under_80ms"] = bool(out["latency_mean_ms"] < 80.0)
        out["H4_fps_at_least_15"] = bool(out["fps_throughput"] >= 15.0)

        stages = {}
        for k, v in self.stage_totals.items():
            stages[k] = {
                "mean_ms": round(float(np.mean(v)), 3),
                "max_ms": round(float(np.max(v)), 3),
            }
        out["stages"] = stages
        return out

    # ------------------------------------------------------------------
    def close(self, extra=None):
        summary = self.summary()
        if extra:
            summary.update(extra)
        if self.dir:
            with open(os.path.join(self.dir, "summary.json"), "w") as fh:
                json.dump(summary, fh, indent=2)
        if self._fh:
            self._fh.close()
            self._fh = None
        return summary


def print_summary(summary, title="RUN SUMMARY"):
    line = "=" * 58
    print("\n" + line)
    print(title)
    print(line)
    order = [
        ("frames", "Frames processed", ""),
        ("duration_s", "Duration", "s"),
        ("fps_throughput", "Throughput (frames/duration)", "fps"),
        ("fps_median", "Median instantaneous rate", "fps"),
        ("fps_min", "Worst frame rate", "fps"),
        ("latency_mean_ms", "Mean end-to-end latency", "ms"),
        ("latency_p95_ms", "95th percentile latency", "ms"),
        ("detection_rate", "Lane detection rate", ""),
        ("confidence_mean", "Mean detection confidence", ""),
        ("mean_abs_error_px", "Mean lateral deviation", "px"),
        ("rms_error_px", "RMS lateral deviation", "px"),
        ("error_std_px", "Deviation std (oscillation)", "px"),
        ("zero_crossings_per_s", "Centre crossings", "1/s"),
    ]
    for key, label, unit in order:
        if key in summary:
            print("  %-30s %10s %s" % (label, summary[key], unit))
    if "stages" in summary and summary["stages"]:
        print("-" * 58)
        print("  Stage timing breakdown (mean ms)")
        for k, v in summary["stages"].items():
            print("    %-24s %8.2f  (max %6.2f)" %
                  (k.replace("t_", ""), v["mean_ms"], v["max_ms"]))
    if summary.get("WARNING"):
        print("-" * 58)
        print("  WARNING: %s" % summary["WARNING"])
    print("-" * 58)
    print("  Frames driven      : %s of %s"
          % (summary.get("frames_armed", "?"),
             summary.get("frames_armed", 0) + summary.get("frames_disarmed", 0)))
    print("  H2 latency < 80 ms : %s" % summary.get("H2_latency_under_80ms"))
    print("  H4 fps >= 15       : %s   (judged on throughput)"
          % summary.get("H4_fps_at_least_15"))
    print(line + "\n")
