"""
camera.py -- Frame acquisition.

Provides one interface (`FrameSource.read() -> (timestamp, frame_bgr)`) with
three backends:

  * PiCameraSource  -- Picamera2 on the Raspberry Pi (CSI bus, low latency)
  * UsbCameraSource -- any V4L2 / USB webcam via OpenCV
  * VideoFileSource -- recorded clip, used for repeatable offline benchmarks

`ThreadedSource` wraps any of them in a grabber thread. This is one of the
"optimisation strategies" the exposé asks about: capture (15-20 ms) then runs
in parallel with processing instead of in series, which is what pushes the
end-to-end latency under the 80 ms target of hypothesis H2.

NOTE on colour order: Picamera2's "RGB888" format actually hands back a
BGR-ordered numpy array, which is exactly what OpenCV expects. The whole
codebase therefore treats every frame as BGR.
"""

import threading
import time

import cv2
import numpy as np


class FrameSource:
    """Base interface."""

    def read(self):
        """Return (capture_timestamp, frame) or (None, None) when exhausted."""
        raise NotImplementedError

    def release(self):
        pass

    @property
    def name(self):
        return self.__class__.__name__


# --------------------------------------------------------------------------
class PiCameraSource(FrameSource):
    def __init__(self, cfg):
        from picamera2 import Picamera2  # imported lazily: Pi only

        self.cfg = cfg
        self.picam2 = Picamera2()
        video_cfg = self.picam2.create_video_configuration(
            main={"format": "RGB888", "size": (cfg.width, cfg.height)},
            controls={"FrameDurationLimits": (
                int(1e6 / cfg.fps_target), int(1e6 / cfg.fps_target))},
        )
        self.picam2.configure(video_cfg)
        self.picam2.start()
        time.sleep(1.0)  # let auto-exposure / AWB settle

    def read(self):
        frame = self.picam2.capture_array()
        return time.perf_counter(), frame

    def release(self):
        try:
            self.picam2.stop()
            self.picam2.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
class UsbCameraSource(FrameSource):
    def __init__(self, cfg):
        self.cap = cv2.VideoCapture(cfg.usb_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps_target)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open USB camera index %d" % cfg.usb_index)
        self.size = (cfg.width, cfg.height)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        return time.perf_counter(), frame

    def release(self):
        self.cap.release()


# --------------------------------------------------------------------------
class VideoFileSource(FrameSource):
    """Replays a recorded clip. Used by benchmark.py for A/B comparison."""

    def __init__(self, cfg, loop=False, realtime=False):
        self.cap = cv2.VideoCapture(cfg.video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video file: %s" % cfg.video_path)
        self.size = (cfg.width, cfg.height)
        self.loop = loop
        self.realtime = realtime
        src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_period = 1.0 / src_fps
        self._next_t = None

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok:
                return None, None
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        if self.realtime:
            now = time.perf_counter()
            if self._next_t is None:
                self._next_t = now
            delay = self._next_t - now
            if delay > 0:
                time.sleep(delay)
            self._next_t += self.frame_period
        return time.perf_counter(), frame

    def release(self):
        self.cap.release()


# --------------------------------------------------------------------------
class ThreadedSource(FrameSource):
    """
    Runs the wrapped source in a background thread and always serves the most
    recent frame. Removes capture blocking from the control loop.
    """

    def __init__(self, source):
        self.source = source
        self._lock = threading.Lock()
        self._frame = None
        self._ts = None
        self._stopped = False
        self._exhausted = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        # wait for the first frame (max 5 s)
        t0 = time.time()
        while self._frame is None and not self._exhausted:
            if time.time() - t0 > 5.0:
                raise RuntimeError("Camera produced no frame within 5 s")
            time.sleep(0.01)

    def _loop(self):
        while not self._stopped:
            ts, frame = self.source.read()
            if frame is None:
                self._exhausted = True
                break
            with self._lock:
                self._frame = frame
                self._ts = ts

    def read(self):
        if self._exhausted and self._frame is None:
            return None, None
        with self._lock:
            if self._frame is None:
                return None, None
            frame, ts = self._frame, self._ts
            if self._exhausted:
                self._frame = None      # serve the last frame exactly once
        return ts, frame.copy()

    def release(self):
        self._stopped = True
        self._thread.join(timeout=1.0)
        self.source.release()

    @property
    def name(self):
        return "Threaded(%s)" % self.source.name


# --------------------------------------------------------------------------
def create_source(cfg, loop=False, realtime=False):
    """Factory driven by CameraConfig.source."""
    if cfg.source == "picamera":
        src = PiCameraSource(cfg)
    elif cfg.source == "usb":
        src = UsbCameraSource(cfg)
    elif cfg.source == "video":
        src = VideoFileSource(cfg, loop=loop, realtime=realtime)
        return src                      # never thread a file: it would skip frames
    else:
        raise ValueError("Unknown camera source: %s" % cfg.source)

    if cfg.threaded:
        src = ThreadedSource(src)
    return src


def orient(frame, cfg):
    """Apply the configured flips."""
    if cfg.flip_horizontal and cfg.flip_vertical:
        return cv2.flip(frame, -1)
    if cfg.flip_horizontal:
        return cv2.flip(frame, 1)
    if cfg.flip_vertical:
        return cv2.flip(frame, 0)
    return frame


def synthetic_frame(width, height, offset=0, curve=0.0, noise=8):
    """
    Generates a synthetic lane image. Used by the self-test so the pipeline can
    be verified without a camera attached.
    """
    img = np.full((height, width, 3), 40, np.uint8)
    lane_half = 80
    for y in range(height):
        t = (height - y) / float(height)
        shift = int(offset + curve * (t ** 2) * width * 0.5)
        cx = width // 2 + shift
        thickness = max(2, int(6 * (1.2 - t)))
        for cxx in (cx - lane_half, cx + lane_half):
            x0, x1 = max(0, cxx - thickness), min(width, cxx + thickness)
            if x0 < x1:
                img[y, x0:x1] = (235, 235, 235)
    if noise:
        img = cv2.add(img, np.random.randint(
            0, noise, img.shape, dtype=np.uint8))
    return img
