"""
dashboard.py -- Real-time telemetry dashboard and digital twin.

Layout (default 320x240 camera frame + 160 px panel = 480x240, upscaled):

    +-----------------------------+----------------+
    |                             |   TELEMETRY    |
    |   camera view with          |   error / cmd  |
    |   lane overlay, detected    |   FPS, latency |
    |   boundaries and centre     |   offset bar   |
    |                             |   P I D bars   |
    |                             |   digital twin |
    +-----------------------------+----------------+

Everything drawn here is diagnostic only. The smoothed error is used for the
display so the numbers are readable; the controller always receives the raw
unsmoothed error.
"""

import cv2
import numpy as np

from graphs import GraphPanel, histogram_strip

# BGR colours
WHITE = (240, 240, 240)
GREY = (150, 150, 150)
DARK = (35, 32, 30)
GREEN = (90, 220, 120)
ORANGE = (40, 140, 250)
RED = (60, 60, 235)
CYAN = (230, 220, 90)
YELLOW = (80, 220, 240)
BLUE = (235, 140, 60)

FONT = cv2.FONT_HERSHEY_SIMPLEX


class Dashboard:
    def __init__(self, cfg, frame_w, frame_h, deadzone_px=6.0):
        self.cfg = cfg
        self.fw, self.fh = frame_w, frame_h
        self.pw = cfg.panel_width
        self.trail = []
        self.total_w = frame_w + cfg.panel_width
        self.graphs = None
        if getattr(cfg, "show_graphs", False):
            self.graphs = GraphPanel(
                self.total_w, height=cfg.graph_height,
                history=cfg.graph_history, deadzone_px=deadzone_px,
                half_width=frame_w / 2.0)

    # ------------------------------------------------------------------
    def render(self, frame, pre, result, state):
        """Returns the composed dashboard image (BGR)."""
        view = frame.copy()
        self._draw_overlay(view, pre, result, state)

        panel = np.full((self.fh, self.pw, 3), DARK, np.uint8)
        self._draw_panel(panel, result, state)

        canvas = np.hstack([view, panel])

        strips = [canvas]
        if self.cfg.show_histogram_strip:
            strips.append(histogram_strip(
                result.histogram, self.total_w,
                peaks=(result.left_x, result.right_x),
                centre=result.lane_center if result.detected else None))
        if self.graphs is not None:
            self.graphs.push(
                error_px=state["error_px"],
                smoothed_px=state["smoothed_error"],
                terms=state["terms"],
                left_duty=state["left_duty"],
                right_duty=state["right_duty"],
                latency_ms=state["latency_ms"],
                fps=state["fps"],
                detected=result.detected)
            strips.append(self.graphs.render())
        if len(strips) > 1:
            canvas = np.vstack(strips)

        s = self.cfg.display_scale
        if s and abs(s - 1.0) > 1e-3:
            canvas = cv2.resize(canvas, None, fx=s, fy=s,
                                interpolation=cv2.INTER_NEAREST)
        return canvas

    # ------------------------------------------------------------------
    def _draw_overlay(self, view, pre, result, state):
        top, bottom = pre.roi_top, pre.roi_bottom
        # ROI band
        band = view[top:bottom, :].copy()
        tint = np.full_like(band, (60, 40, 20))
        view[top:bottom, :] = cv2.addWeighted(band, 0.75, tint, 0.25, 0)
        cv2.line(view, (0, top), (self.fw, top), (90, 90, 90), 1)

        # target centre
        cx = self.fw // 2
        cv2.line(view, (cx, top), (cx, self.fh), GREY, 1)

        if result.lines:
            for (x1, y1, x2, y2) in result.lines:
                cv2.line(view, (x1, y1 + top), (x2, y2 + top), (200, 200, 60), 1)

        if result.detected:
            for x, col in ((result.left_x, BLUE), (result.right_x, BLUE)):
                if x is not None:
                    xi = int(np.clip(x, 0, self.fw - 1))
                    cv2.line(view, (xi, top), (xi, self.fh - 1), col, 2)
            lc = int(np.clip(result.lane_center, 0, self.fw - 1))
            cv2.circle(view, (lc, (top + self.fh) // 2), 6,
                       GREEN if not result.inferred else YELLOW, -1)
            cv2.arrowedLine(view, (cx, self.fh - 6), (lc, self.fh - 6),
                            GREEN, 2, tipLength=0.35)
        else:
            cv2.putText(view, "NO LANE", (self.fw // 2 - 40, self.fh // 2),
                        FONT, 0.5, RED, 2, cv2.LINE_AA)

        # histogram sparkline along the bottom of the ROI
        if result.histogram is not None:
            h = result.histogram.astype(np.float32)
            peak = float(h.max()) if h.max() > 0 else 1.0
            hgt = max(10, (bottom - top) - 4)
            pts = []
            for x in range(0, self.fw, 2):
                y = int(self.fh - 2 - (h[x] / peak) * hgt)
                pts.append((x, y))
            if len(pts) > 1:
                cv2.polylines(view, [np.array(pts, np.int32)], False, CYAN, 1)

    # ------------------------------------------------------------------
    def _draw_panel(self, p, result, state):
        pw = self.pw
        compact = self.graphs is not None   # PID bars live in the charts instead

        y = 16
        cv2.putText(p, "TELEMETRY", (10, y), FONT, 0.42, WHITE, 1, cv2.LINE_AA)
        y += 6
        cv2.line(p, (10, y), (pw - 10, y), (90, 85, 80), 1)

        y += 16
        cv2.putText(p, "ALGO %s" % state["algo"][:9].upper(),
                    (10, y), FONT, 0.33, CYAN, 1, cv2.LINE_AA)
        y += 14
        cv2.putText(p, "CTRL %s" % state["controller"].upper(),
                    (10, y), FONT, 0.33, CYAN, 1, cv2.LINE_AA)

        y += 18
        err = state["smoothed_error"]
        cv2.putText(p, "Error: %+.0f px" % err, (10, y), FONT, 0.38,
                    YELLOW, 1, cv2.LINE_AA)

        y += 16
        cmd = state["command"]
        col = GREEN if cmd == "STRAIGHT" else ORANGE
        cv2.putText(p, "CMD: %s" % cmd, (10, y), FONT, 0.38, col, 1, cv2.LINE_AA)

        # ---- offset bar ------------------------------------------------
        y += 12
        bx0, bx1 = 20, pw - 20
        cv2.rectangle(p, (bx0, y), (bx1, y + 8), (70, 66, 62), -1)
        mid = (bx0 + bx1) // 2
        cv2.line(p, (mid, y - 2), (mid, y + 10), GREY, 1)
        dz = state["deadzone_px"]
        half = self.fw / 2.0
        dz_px = int((dz / half) * (bx1 - bx0) / 2)
        cv2.rectangle(p, (mid - dz_px, y), (mid + dz_px, y + 8), (60, 90, 60), -1)
        pos = int(mid + np.clip(err / half, -1, 1) * (bx1 - bx0) / 2)
        cv2.circle(p, (pos, y + 4), 5, RED, -1)
        y += 14

        # ---- PID term bars (only when the live charts are switched off) --
        if not compact:
            y += 10
            cv2.putText(p, "PID TERMS", (10, y), FONT, 0.32, GREY, 1, cv2.LINE_AA)
            for label, val, c in zip("PID", state["terms"], (GREEN, YELLOW, ORANGE)):
                y += 12
                cv2.putText(p, label, (10, y + 4), FONT, 0.32, WHITE, 1, cv2.LINE_AA)
                base, span = 30, pw - 45 - 30
                zero = base + span // 2
                cv2.line(p, (base, y + 1), (base + span, y + 1), (70, 66, 62), 3)
                end = int(zero + np.clip(val, -1, 1) * span / 2)
                cv2.line(p, (zero, y + 1), (end, y + 1), c, 3)
            y += 8

        # ---- digital twin ----------------------------------------------
        twin_top = y + 14
        twin_bottom = self.fh - 40
        if twin_bottom - twin_top > 40:
            self._draw_twin(p, err, twin_top, twin_bottom)

        # ---- performance and motors ------------------------------------
        lat_ok = state["latency_ms"] < 80
        cv2.putText(p, "%.1f FPS" % state["fps"], (10, self.fh - 28),
                    FONT, 0.33, WHITE, 1, cv2.LINE_AA)
        cv2.putText(p, "%.0f ms" % state["latency_ms"], (72, self.fh - 28),
                    FONT, 0.33, GREEN if lat_ok else RED, 1, cv2.LINE_AA)
        cv2.putText(p, "conf %.2f  w %.0f" % (result.confidence,
                                             result.line_width),
                    (10, self.fh - 16), FONT, 0.32, GREY, 1, cv2.LINE_AA)
        cv2.putText(p, "L %3.0f%%  R %3.0f%%" % (state["left_duty"], state["right_duty"]),
                    (10, self.fh - 4), FONT, 0.33,
                    WHITE if state["motors_armed"] else GREY, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    def _draw_twin(self, p, err, top, bottom):
        pw = self.pw
        lx, rx = 30, pw - 30
        cv2.line(p, (lx, top), (lx, bottom), (120, 118, 115), 2)
        cv2.line(p, (rx, top), (rx, bottom), (120, 118, 115), 2)
        cv2.putText(p, "DIGITAL TWIN", (10, top - 3), FONT, 0.30, GREY, 1, cv2.LINE_AA)

        car_x = int(np.clip((lx + rx) / 2 + err * 0.4, lx + 14, rx - 14))
        car_y = (top + bottom) // 2

        self.trail.append(car_x)
        if len(self.trail) > 40:
            self.trail.pop(0)
        for i, tx in enumerate(self.trail):
            ty = bottom - int((len(self.trail) - i) * 1.4)
            if top < ty < bottom:
                cv2.circle(p, (int(tx), ty), 1, (80, 76, 72), -1)

        cv2.rectangle(p, (car_x - 12, car_y - 15), (car_x + 12, car_y + 15),
                      (40, 80, 235), -1)
        cv2.rectangle(p, (car_x - 9, car_y - 11), (car_x + 9, car_y - 3),
                      CYAN, -1)
        for dx in (-14, 14):
            cv2.rectangle(p, (car_x + dx - 2, car_y - 10),
                          (car_x + dx + 2, car_y + 10), (200, 200, 200), -1)


# ----------------------------------------------------------------------
def debug_windows(pre, result):
    """Optional extra windows for the vision chapter screenshots."""
    cv2.imshow("1 gray", pre.gray)
    cv2.imshow("2 blurred", pre.blurred)
    cv2.imshow("3 binary", pre.binary)
    cv2.imshow("4 canny", pre.edges)
    if result.histogram is not None:
        h = result.histogram.astype(np.float32)
        peak = max(1.0, float(h.max()))
        plot = np.zeros((120, len(h), 3), np.uint8)
        for x in range(len(h)):
            v = int((h[x] / peak) * 118)
            cv2.line(plot, (x, 119), (x, 119 - v), CYAN, 1)
        cv2.imshow("5 histogram", plot)
