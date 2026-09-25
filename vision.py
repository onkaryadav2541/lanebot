"""
vision.py -- Preprocessing stage of the perception pipeline.

Implements exactly the chain described in the exposé, section 3:
grayscale -> Gaussian blur -> thresholding -> Canny -> ROI mask.

Three thresholding strategies are implemented so that hypothesis H3
("dynamic thresholding is essential for robust lane detection under variable
lighting") can be tested rather than assumed:

    fixed     -- constant threshold (the research-paper baseline, T = 140)
    otsu      -- global threshold recomputed per frame from the histogram
    adaptive  -- per-region Gaussian threshold, handles shadows across the frame

Every stage is timed individually so that the thesis can reproduce the
frame-processing timing table with measured, not estimated, numbers.
"""

from dataclasses import dataclass, field
import time

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    gray: np.ndarray = None
    blurred: np.ndarray = None
    binary: np.ndarray = None
    edges: np.ndarray = None
    roi_binary: np.ndarray = None
    roi_edges: np.ndarray = None
    roi_top: int = 0
    roi_bottom: int = 0
    roi_fill: float = 0.0        # fraction of the ROI occupied by lane mask
    threshold_used: float = 0.0
    timings: dict = field(default_factory=dict)


class Preprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self._clahe = None
        if cfg.use_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=cfg.clahe_clip, tileGridSize=(8, 8))

    # ------------------------------------------------------------------
    def _threshold(self, blurred):
        c = self.cfg
        style = cv2.THRESH_BINARY_INV if c.invert_binary else cv2.THRESH_BINARY

        if c.threshold_mode == "fixed":
            used, binary = cv2.threshold(
                blurred, c.fixed_threshold, 255, style)
            return float(c.fixed_threshold), binary

        if c.threshold_mode == "otsu":
            used, binary = cv2.threshold(
                blurred, 0, 255, style | cv2.THRESH_OTSU)
            return float(used), binary

        if c.threshold_mode == "adaptive":
            block = c.adaptive_block if c.adaptive_block % 2 == 1 else c.adaptive_block + 1
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                style, block, c.adaptive_c)
            return float(np.mean(blurred)), binary

        raise ValueError("Unknown threshold_mode: %s" % c.threshold_mode)

    # ------------------------------------------------------------------
    def process(self, frame):
        """frame: BGR uint8 -> PreprocessResult"""
        c = self.cfg
        t = {}
        h, w = frame.shape[:2]

        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._clahe is not None:
            gray = self._clahe.apply(gray)
        t["grayscale"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        k = c.blur_kernel if c.blur_kernel % 2 == 1 else c.blur_kernel + 1
        blurred = cv2.GaussianBlur(gray, (k, k), 0)
        t["blur"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        thr_used, binary = self._threshold(blurred)
        t["threshold"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        edges = cv2.Canny(blurred, c.canny_low, c.canny_high)
        t["canny"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        top = int(h * c.roi_top_ratio)
        bottom = int(h * c.roi_bottom_ratio)
        top = max(0, min(top, h - 2))
        bottom = max(top + 1, min(bottom, h))
        roi_binary = binary[top:bottom, :]
        roi_edges = edges[top:bottom, :]
        t["roi"] = (time.perf_counter() - t0) * 1000

        # Fraction of the ROI that the lane mask covers. Two lane markings
        # cover only a small part of it; a band painted across the full width
        # of the lane covers most of it. That difference is what makes a
        # stop marker detectable without any additional machinery.
        roi_fill = float(np.count_nonzero(roi_binary)) / float(roi_binary.size)

        return PreprocessResult(
            gray=gray, blurred=blurred, binary=binary, edges=edges,
            roi_binary=roi_binary, roi_edges=roi_edges,
            roi_top=top, roi_bottom=bottom, roi_fill=roi_fill,
            threshold_used=thr_used, timings=t,
        )

    # ------------------------------------------------------------------
    def roi_for_histogram(self, pre):
        """Select which mask the histogram detector should sum over."""
        mode = self.cfg.hist_source
        if mode == "binary":
            return pre.roi_binary
        if mode == "edges":
            return pre.roi_edges
        if mode == "combined":
            return cv2.bitwise_or(pre.roi_binary, pre.roi_edges)
        raise ValueError("Unknown hist_source: %s" % mode)
