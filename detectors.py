"""
detectors.py -- The two lane-detection algorithms compared in the exposé.

    Algorithm A : Canny edge detection + probabilistic Hough transform
    Algorithm B : Histogram (column-sum) peak detection

Both expose the same interface

    detector.detect(pre, width) -> LaneResult

so that main.py can hot-swap them at runtime (press 'a') and benchmark.py can
run both over the identical recorded footage -- which is what makes the
comparison in the thesis fair.

Robustness features shared by both detectors:
  * single-boundary recovery: if only one lane line is visible, the opposite
    boundary is inferred from the online estimate of the lane width, instead
    of throwing the frame away.
  * online lane-width estimation: the expected width adapts slowly to the
    track actually being driven.
  * confidence value in [0, 1], used by the controller and logged for the
    robustness evaluation.
"""

from dataclasses import dataclass
import time

import cv2
import numpy as np


@dataclass
class LaneResult:
    detected: bool = False
    lane_center: float = 0.0
    left_x: float = None
    right_x: float = None
    confidence: float = 0.0
    inferred: bool = False        # True if one boundary had to be estimated
    histogram: np.ndarray = None  # for the dashboard / debug windows
    lines: list = None            # Hough segments, for the overlay
    line_width: float = 0.0       # mean detected marking thickness, px
    detect_ms: float = 0.0


class BaseDetector:
    name = "base"

    def __init__(self, cfg):
        self.cfg = cfg
        self.lane_width = float(cfg.expected_lane_width)

    # -- shared helpers -------------------------------------------------
    def _update_width(self, left_x, right_x):
        width = right_x - left_x
        if 0.35 * self.cfg.expected_lane_width < width < 2.2 * self.cfg.expected_lane_width:
            a = self.cfg.lane_width_alpha
            self.lane_width = (1 - a) * self.lane_width + a * width

    def _combine(self, left_x, right_x, conf_l, conf_r, width):
        """Turn zero/one/two detected boundaries into a lane centre."""
        inferred = False
        if left_x is not None and right_x is not None:
            if right_x - left_x < 0.30 * self.cfg.expected_lane_width:
                # the two "boundaries" collapsed onto the same marking
                left_x, right_x = (left_x, None) if conf_l >= conf_r else (None, right_x)
            else:
                self._update_width(left_x, right_x)

        if left_x is not None and right_x is not None:
            # Plausibility gate: a pair of boundaries whose separation is
            # nothing like the known lane width is not the lane. This is what
            # stops the detector reporting the edges of the sheet, or the
            # image border, as a confident detection.
            tol = getattr(self.cfg, "lane_width_tolerance", None)
            if tol is not None:
                w = right_x - left_x
                lo = self.cfg.expected_lane_width * (1.0 - tol)
                hi = self.cfg.expected_lane_width * (1.0 + tol)
                if not (lo <= w <= hi):
                    return LaneResult(detected=False)
            center = 0.5 * (left_x + right_x)
            conf = min(conf_l, conf_r)
        elif left_x is not None:
            center = left_x + self.lane_width / 2.0
            conf, inferred = conf_l * 0.6, True
        elif right_x is not None:
            center = right_x - self.lane_width / 2.0
            conf, inferred = conf_r * 0.6, True
        else:
            return LaneResult(detected=False)

        center = float(np.clip(center, 0, width - 1))
        return LaneResult(detected=True, lane_center=center,
                          left_x=left_x, right_x=right_x,
                          confidence=float(np.clip(conf, 0.0, 1.0)),
                          inferred=inferred)


# ======================================================================
class HistogramDetector(BaseDetector):
    """
    Algorithm B -- sums the mask vertically over the ROI and looks for the
    strongest column in the left and right half of the image.

    Complexity is O(N) in the number of ROI pixels and the aggregation over
    many rows makes it tolerant of reflections and paper texture, which is
    why the exposé expects it to handle curves more smoothly (H1).
    """

    name = "histogram"

    def __init__(self, cfg, mask_selector):
        super().__init__(cfg)
        self.mask_selector = mask_selector   # callable(pre) -> roi mask

    @staticmethod
    def _measure_thickness(roi_binary, res, width):
        """Mean width, in columns, of the markings at the detected boundaries."""
        col = np.count_nonzero(roi_binary, axis=0).astype(np.float32)
        if col.max() <= 0:
            return 0.0
        floor = 0.35 * float(col.max())
        widths = []
        for x in (res.left_x, res.right_x):
            if x is None:
                continue
            xi = int(np.clip(round(x), 0, width - 1))
            if col[xi] < floor:
                # The boundary position comes from the Canny edge, which sits
                # on the rim of the marking rather than inside it. Step to the
                # nearest filled column before measuring the run.
                seed = None
                for d in range(1, 14):
                    for cand in (xi - d, xi + d):
                        if 0 <= cand < width and col[cand] >= floor:
                            seed = cand
                            break
                    if seed is not None:
                        break
                if seed is None:
                    continue
                xi = seed
            lo = xi
            while lo > 0 and col[lo - 1] >= floor and xi - lo < 40:
                lo -= 1
            hi = xi
            while hi < width - 1 and col[hi + 1] >= floor and hi - xi < 40:
                hi += 1
            widths.append(float(hi - lo + 1))
        return float(np.mean(widths)) if widths else 0.0

    @staticmethod
    def _peak_width(hist, peak_idx, half_window=24):
        """
        Thickness of the marking at this peak, in columns.

        Measured as the run of columns around the peak that stay above half
        its height. A thin lane line gives a narrow run and a thicker one a
        wide run, which is what lets a deliberately widened section of tape be
        recognised without changing anything else about the track: the lane
        still looks like a lane.
        """
        peak = float(hist[peak_idx])
        if peak <= 0:
            return 0.0
        half = 0.5 * peak
        lo = peak_idx
        while lo > 0 and lo > peak_idx - half_window and hist[lo - 1] >= half:
            lo -= 1
        hi = peak_idx
        n = len(hist)
        while hi < n - 1 and hi < peak_idx + half_window and hist[hi + 1] >= half:
            hi += 1
        return float(hi - lo + 1)

    @staticmethod
    def _refine(hist, peak_idx, half_window=14):
        """
        Sub-pixel peak position.

        np.argmax alone is biased: a lane marking produces two Canny edges and
        argmax always returns the first of two equal columns, which pulls both
        peaks to the left and shifts the computed lane centre by several
        pixels. Taking the intensity-weighted centroid of the contiguous
        region around the peak removes that bias and additionally gives
        sub-pixel resolution, which matters because the derivative term of the
        PID is computed from this signal.
        """
        lo = max(0, peak_idx - half_window)
        hi = min(len(hist), peak_idx + half_window + 1)
        window = hist[lo:hi].astype(np.float64)
        floor = 0.35 * float(hist[peak_idx])
        weights = np.clip(window - floor, 0.0, None)
        if weights.sum() <= 0:
            return float(peak_idx)
        idx = np.arange(lo, hi, dtype=np.float64)
        return float(np.sum(idx * weights) / np.sum(weights))

    def detect(self, pre, width):
        t0 = time.perf_counter()
        c = self.cfg
        roi = self.mask_selector(pre)
        rows = roi.shape[0]

        hist = np.sum(roi, axis=0, dtype=np.int32)

        if c.hist_smooth > 1:
            k = np.ones(c.hist_smooth, dtype=np.float32) / c.hist_smooth
            hist = np.convolve(hist, k, mode="same").astype(np.int32)

        m = c.hist_margin
        center_col = width // 2
        left_slice = hist[m:center_col]
        right_slice = hist[center_col:width - m]

        left_x = right_x = None
        conf_l = conf_r = 0.0
        widths = []
        max_possible = float(rows * 255)

        if left_slice.size:
            i = int(np.argmax(left_slice))
            if left_slice[i] >= c.hist_min_peak:
                left_x = self._refine(hist, i + m)
                conf_l = float(left_slice[i]) / max_possible
                widths.append(self._peak_width(hist, i + m))
        if right_slice.size:
            i = int(np.argmax(right_slice))
            if right_slice[i] >= c.hist_min_peak:
                right_x = self._refine(hist, i + center_col)
                conf_r = float(right_slice[i]) / max_possible
                widths.append(self._peak_width(hist, i + center_col))

        res = self._combine(left_x, right_x, conf_l, conf_r, width)
        res.histogram = hist

        # Marking thickness is measured on the binary mask, never on the Canny
        # output. Canny reports the two edges of a marking, and those edges are
        # a fixed couple of pixels wide however thick the tape is, so an edge
        # histogram cannot distinguish a thin line from a wide one.
        res.line_width = self._measure_thickness(
            pre.roi_binary, res, width) if res.detected else 0.0
        res.detect_ms = (time.perf_counter() - t0) * 1000
        return res


# ======================================================================
class CannyHoughDetector(BaseDetector):
    """
    Algorithm A -- probabilistic Hough transform on the Canny edge image.

    Segments are classified into the left and right boundary by the sign of
    their slope combined with their position relative to the image centre.
    Each side is then fitted with a length-weighted first-order polynomial
    x = m*y + b, evaluated at the bottom row of the ROI, i.e. the point
    closest to the vehicle.
    """

    name = "canny_hough"

    def detect(self, pre, width):
        t0 = time.perf_counter()
        c = self.cfg
        roi = pre.roi_edges
        rows = roi.shape[0]
        center_col = width / 2.0

        segments = cv2.HoughLinesP(
            roi, rho=1, theta=np.pi / 180,
            threshold=c.hough_threshold,
            minLineLength=c.hough_min_line_len,
            maxLineGap=c.hough_max_line_gap)

        left_pts, left_w, right_pts, right_w = [], [], [], []
        drawn = []

        if segments is not None:
            for seg in segments:
                x1, y1, x2, y2 = seg[0]
                drawn.append((int(x1), int(y1), int(x2), int(y2)))
                dx, dy = float(x2 - x1), float(y2 - y1)
                if abs(dx) < 1e-3:
                    slope = 1e3
                else:
                    slope = dy / dx
                if abs(slope) < c.min_abs_slope:
                    continue                      # near-horizontal -> not a lane
                length = float(np.hypot(dx, dy))
                mid_x = 0.5 * (x1 + x2)
                if  mid_x < center_col:
                    left_pts += [(y1, x1), (y2, x2)]
                    left_w += [length, length]
                elif mid_x > center_col:
                    right_pts += [(y1, x1), (y2, x2)]
                    right_w += [length, length]

        y_eval = rows // 2

        def fit(points, weights):
            if len(points) < 2:
                return None, 0.0
            ys = np.array([p[0] for p in points], dtype=np.float64)
            xs = np.array([p[1] for p in points], dtype=np.float64)
            ws = np.array(weights, dtype=np.float64)
            if np.ptp(ys) < 1.0:                  # all points on one row
                return float(np.average(xs, weights=ws)), min(1.0, len(points) / 8.0)
            m, b = np.polyfit(ys, xs, 1, w=ws)
            return float(m * y_eval + b), min(1.0, len(points) / 8.0)

        left_x, conf_l = fit(left_pts, left_w)
        right_x, conf_r = fit(right_pts, right_w)

        if left_x is not None and not (-width < left_x < 2 * width):
            left_x, conf_l = None, 0.0
        if right_x is not None and not (-width < right_x < 2 * width):
            right_x, conf_r = None, 0.0

        res = self._combine(left_x, right_x, conf_l, conf_r, width)
        res.lines = drawn
        res.detect_ms = (time.perf_counter() - t0) * 1000
        return res


# ======================================================================
def create_detector(name, vision_cfg, preprocessor):
    if name in ("histogram", "b", "B"):
        return HistogramDetector(vision_cfg, preprocessor.roi_for_histogram)
    if name in ("canny_hough", "hough", "a", "A"):
        return CannyHoughDetector(vision_cfg)
    raise ValueError("Unknown algorithm: %s" % name)
