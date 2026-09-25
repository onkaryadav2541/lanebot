"""
graphs.py -- Live scrolling charts for the telemetry dashboard.

Everything here is drawn with plain OpenCV primitives into a numpy array
rather than with matplotlib, because matplotlib cannot redraw at 15 Hz on a
Raspberry Pi 3B+. The cost of the whole graph strip is a few milliseconds,
and it can be throttled with DashboardConfig.update_every or switched off
entirely with show_graphs.

These charts are for driving and debugging. For the figures that go into the
thesis use report.py, which reads frames.csv afterwards and produces proper
vector-quality plots -- a screenshot of a 15 Hz OpenCV window looks amateur
in a printed document.
"""

from collections import deque

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (28, 26, 24)
GRID = (58, 55, 52)
AXIS = (110, 106, 102)
TEXT = (215, 212, 208)


class Series:
    """A fixed-length ring buffer of samples."""

    def __init__(self, length, colour, label):
        self.data = deque(maxlen=length)
        self.colour = colour
        self.label = label

    def push(self, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = np.nan
        self.data.append(v)


class MiniChart:
    """
    One small line chart drawn into a region of an image.

    y_range = None  -> autoscale on the visible window
    y_range = (a,b) -> fixed limits
    """

    def __init__(self, title, series, y_range=None, zero_line=True,
                 band=None, unit="", min_span=None, symmetric=False):
        self.title = title
        self.series = series
        self.y_range = y_range
        self.min_span = min_span
        self.symmetric = symmetric
        self.zero_line = zero_line
        self.band = band            # (lo, hi) shaded reference band
        self.unit = unit

    # ------------------------------------------------------------------
    def _limits(self):
        if self.y_range:
            return self.y_range
        vals = []
        for s in self.series:
            vals.extend([v for v in s.data if np.isfinite(v)])
        if not vals:
            return -1.0, 1.0
        lo, hi = float(min(vals)), float(max(vals))
        if abs(hi - lo) < 1e-6:
            lo, hi = lo - 1.0, hi + 1.0
        pad = 0.12 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        if self.symmetric:
            m = max(abs(lo), abs(hi))
            lo, hi = -m, m
        if self.min_span and (hi - lo) < self.min_span:
            mid = 0.5 * (lo + hi)
            lo, hi = mid - self.min_span / 2.0, mid + self.min_span / 2.0
        return lo, hi

    # ------------------------------------------------------------------
    def draw(self, canvas, x, y, w, h):
        roi = canvas[y:y + h, x:x + w]
        roi[:] = BG
        cv2.rectangle(roi, (0, 0), (w - 1, h - 1), GRID, 1)

        pad_l, pad_t, pad_b = 30, 14, 10
        pw = w - pad_l - 4
        ph = h - pad_t - pad_b
        if pw < 10 or ph < 10:
            return

        lo, hi = self._limits()
        span = hi - lo

        def ypix(v):
            return int(pad_t + (hi - v) / span * ph)

        # reference band (e.g. the control deadzone, or the 80 ms target)
        if self.band:
            b0, b1 = ypix(self.band[1]), ypix(self.band[0])
            b0, b1 = max(pad_t, min(b0, b1)), min(pad_t + ph, max(b0, b1))
            if b1 > b0:
                sub = roi[b0:b1, pad_l:pad_l + pw]
                sub[:] = cv2.addWeighted(sub, 0.6,
                                         np.full_like(sub, (45, 70, 45)), 0.4, 0)

        # grid
        for frac in (0.0, 0.5, 1.0):
            gy = int(pad_t + frac * ph)
            cv2.line(roi, (pad_l, gy), (pad_l + pw, gy), GRID, 1)
        if self.zero_line and lo < 0 < hi:
            zy = ypix(0.0)
            cv2.line(roi, (pad_l, zy), (pad_l + pw, zy), AXIS, 1)

        # axis labels
        cv2.putText(roi, "%.0f" % hi, (2, pad_t + 4), FONT, 0.28, AXIS, 1, cv2.LINE_AA)
        cv2.putText(roi, "%.0f" % lo, (2, pad_t + ph), FONT, 0.28, AXIS, 1, cv2.LINE_AA)

        # traces
        for s in self.series:
            n = len(s.data)
            if n < 2:
                continue
            step = pw / float(max(1, s.data.maxlen - 1))
            start = pw - (n - 1) * step
            pts = []
            for i, v in enumerate(s.data):
                if not np.isfinite(v):
                    continue
                px = int(pad_l + start + i * step)
                py = int(np.clip(ypix(v), pad_t, pad_t + ph))
                pts.append((px, py))
            if len(pts) > 1:
                cv2.polylines(roi, [np.array(pts, np.int32)], False,
                              s.colour, 1, cv2.LINE_AA)
            if pts:
                cv2.circle(roi, pts[-1], 2, s.colour, -1)

        # title and legend
        cv2.putText(roi, self.title, (pad_l, 10), FONT, 0.31, TEXT, 1, cv2.LINE_AA)
        lx = pad_l + 6 * len(self.title) + 10
        for s in self.series:
            if not s.label or lx + 24 + 6 * len(s.label) > w - 4:
                continue
            cv2.line(roi, (lx, 7), (lx + 8, 7), s.colour, 2)
            cv2.putText(roi, s.label, (lx + 11, 10), FONT, 0.28,
                        s.colour, 1, cv2.LINE_AA)
            lx += 20 + 6 * len(s.label)

        # current value
        last = None
        for s in self.series:
            if s.data and np.isfinite(s.data[-1]):
                last = s.data[-1]
                break
        if last is not None:
            txt = "%.1f%s" % (last, self.unit)
            cv2.putText(roi, txt, (w - 8 - 7 * len(txt), h - 2),
                        FONT, 0.30, TEXT, 1, cv2.LINE_AA)


# ----------------------------------------------------------------------
class GraphPanel:
    """
    The whole strip of live charts drawn under the camera view.

    Layout (for a 480 px wide canvas):

        +--------------------------------------------------+
        |  lateral error (px)  full width                  |
        +----------------+----------------+----------------+
        |  PID terms     |  motor duty %  |  latency / fps |
        +----------------+----------------+----------------+
    """

    COL_ERR = (90, 220, 250)
    COL_SMOOTH = (200, 160, 90)
    COL_P = (120, 230, 140)
    COL_I = (90, 210, 240)
    COL_D = (60, 140, 250)
    COL_L = (240, 180, 90)
    COL_R = (150, 120, 240)
    COL_LAT = (90, 130, 250)
    COL_FPS = (120, 230, 140)

    def __init__(self, width, height=190, history=220, deadzone_px=6.0,
                 half_width=160.0):
        self.w = width
        self.h = height
        self.half_width = half_width

        self.s_err = Series(history, self.COL_ERR, "raw")
        self.s_smooth = Series(history, self.COL_SMOOTH, "smooth")
        self.s_p = Series(history, self.COL_P, "P")
        self.s_i = Series(history, self.COL_I, "I")
        self.s_d = Series(history, self.COL_D, "D")
        self.s_left = Series(history, self.COL_L, "L")
        self.s_right = Series(history, self.COL_R, "R")
        self.s_lat = Series(history, self.COL_LAT, "ms")
        self.s_fps = Series(history, self.COL_FPS, "fps")

        # autoscaling with a floor: a flat trace on a fixed +-160 px axis
        # hides exactly the small oscillations that matter for tuning
        self.chart_error = MiniChart(
            "LATERAL ERROR", [self.s_err, self.s_smooth],
            band=(-deadzone_px, deadzone_px), unit=" px",
            min_span=60.0, symmetric=True)
        self.chart_pid = MiniChart(
            "PID", [self.s_p, self.s_i, self.s_d],
            min_span=0.4, symmetric=True)
        self.chart_motor = MiniChart(
            "DUTY", [self.s_left, self.s_right],
            y_range=(0, 105), zero_line=False, unit=" %")
        self.chart_perf = MiniChart(
            "LATENCY", [self.s_lat, self.s_fps],
            y_range=(0, 140), zero_line=False, band=(0, 80), unit=" ms")

    # ------------------------------------------------------------------
    def push(self, error_px, smoothed_px, terms, left_duty, right_duty,
             latency_ms, fps, detected=True):
        self.s_err.push(error_px if detected else np.nan)
        self.s_smooth.push(smoothed_px)
        self.s_p.push(terms[0])
        self.s_i.push(terms[1])
        self.s_d.push(terms[2])
        self.s_left.push(left_duty)
        self.s_right.push(right_duty)
        self.s_lat.push(latency_ms)
        self.s_fps.push(fps)

    # ------------------------------------------------------------------
    def render(self):
        canvas = np.full((self.h, self.w, 3), BG, np.uint8)
        top_h = int(self.h * 0.46)
        bot_h = self.h - top_h
        third = self.w // 3

        self.chart_error.draw(canvas, 0, 0, self.w, top_h)
        self.chart_pid.draw(canvas, 0, top_h, third, bot_h)
        self.chart_motor.draw(canvas, third, top_h, third, bot_h)
        self.chart_perf.draw(canvas, 2 * third, top_h,
                             self.w - 2 * third, bot_h)
        return canvas


# ----------------------------------------------------------------------
def histogram_strip(hist, width, height=64, peaks=(None, None), centre=None):
    """A proper bar plot of the lane histogram, for the dashboard."""
    img = np.full((height, width, 3), BG, np.uint8)
    if hist is None:
        return img
    h = np.asarray(hist, dtype=np.float32)
    peak = max(1.0, float(h.max()))
    for x in range(min(width, len(h))):
        v = int((h[x] / peak) * (height - 12))
        cv2.line(img, (x, height - 1), (x, height - 1 - v), (110, 190, 210), 1)
    for px, col in zip(peaks, ((235, 140, 60), (235, 140, 60))):
        if px is not None:
            xi = int(np.clip(px, 0, width - 1))
            cv2.line(img, (xi, 0), (xi, height - 1), col, 1)
    if centre is not None:
        xi = int(np.clip(centre, 0, width - 1))
        cv2.line(img, (xi, 0), (xi, height - 1), (120, 230, 140), 1)
    cv2.line(img, (width // 2, 0), (width // 2, height - 1), AXIS, 1)
    cv2.putText(img, "LANE HISTOGRAM  peak=%.0f" % peak, (4, 10),
                FONT, 0.30, TEXT, 1, cv2.LINE_AA)
    return img
