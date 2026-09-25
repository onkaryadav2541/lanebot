#!/usr/bin/env python3
"""
benchmark.py -- Offline comparison of the two lane-detection algorithms.

Runs Algorithm A (Canny + Hough) and Algorithm B (histogram) over *identical*
footage, optionally across all three thresholding modes, and reports the
metrics the exposé asks for: FPS, per-frame latency, detection rate and
steering stability. Because both algorithms see exactly the same frames, the
comparison is fair in a way that two separate live runs never are.

Usage:
    python3 benchmark.py --video runs/run_.../raw.avi
    python3 benchmark.py --video road.avi --thresholds       # sweep H3
    python3 benchmark.py --synthetic 400                     # no footage needed

Outputs (in results/benchmark_<timestamp>/):
    comparison.csv      one row per (algorithm, threshold) combination
    comparison.md       the same table, ready to paste into the thesis
    per_frame.csv       every frame of every configuration
    plots/*.png         error traces, latency distribution, FPS bars
"""

import argparse
import csv
import json
import os
import time

import cv2
import numpy as np

import camera as cam
from config import Config
from detectors import create_detector
from vision import Preprocessor

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


# ----------------------------------------------------------------------
def load_frames(args, cfg):
    """Returns a list of BGR frames, either from a video or synthesised."""
    frames = []
    if args.video:
        capture = cv2.VideoCapture(args.video)
        if not capture.isOpened():
            raise SystemExit("Cannot open video: %s" % args.video)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != (cfg.camera.width, cfg.camera.height):
                frame = cv2.resize(frame, (cfg.camera.width, cfg.camera.height))
            frames.append(frame)
            if args.max_frames and len(frames) >= args.max_frames:
                break
        capture.release()
    elif args.images:
        for name in sorted(os.listdir(args.images)):
            path = os.path.join(args.images, name)
            img = cv2.imread(path)
            if img is None:
                continue
            frames.append(cv2.resize(img, (cfg.camera.width, cfg.camera.height)))
    else:
        n = args.synthetic or 300
        for i in range(n):
            phase = 2 * np.pi * i / max(1, n / 3.0)
            offset = 45 * np.sin(phase)
            curve = 0.35 * np.sin(phase / 2.0)
            frames.append(cam.synthetic_frame(
                cfg.camera.width, cfg.camera.height,
                offset=offset, curve=curve, noise=args.noise))
    if not frames:
        raise SystemExit("No frames loaded")
    return frames


# ----------------------------------------------------------------------
def run_configuration(frames, cfg, algorithm, threshold_mode, hist_source=None):
    cfg.vision.threshold_mode = threshold_mode
    if hist_source:
        cfg.vision.hist_source = hist_source
    pre = Preprocessor(cfg.vision)
    detector = create_detector(algorithm, cfg.vision, pre)

    target = cfg.camera.width / 2.0
    rows = []
    for idx, frame in enumerate(frames):
        t0 = time.perf_counter()
        p = pre.process(frame)
        result = detector.detect(p, cfg.camera.width)
        total_ms = (time.perf_counter() - t0) * 1000.0
        rows.append({
            "algorithm": algorithm,
            "threshold": threshold_mode,
            "hist_source": cfg.vision.hist_source,
            "frame": idx,
            "detected": int(result.detected),
            "inferred": int(result.inferred),
            "confidence": round(result.confidence, 4),
            "lane_center": round(result.lane_center, 2),
            "error_px": round(result.lane_center - target, 2) if result.detected else "",
            "detect_ms": round(result.detect_ms, 3),
            "total_ms": round(total_ms, 3),
            "t_blur": round(p.timings["blur"], 3),
            "t_threshold": round(p.timings["threshold"], 3),
            "t_canny": round(p.timings["canny"], 3),
        })
    return rows


def summarise(rows):
    det = np.array([r["detected"] for r in rows], dtype=float)
    total = np.array([r["total_ms"] for r in rows], dtype=float)
    detect = np.array([r["detect_ms"] for r in rows], dtype=float)
    errs = np.array([r["error_px"] for r in rows if r["error_px"] != ""], dtype=float)
    conf = np.array([r["confidence"] for r in rows], dtype=float)

    if errs.size > 1:
        jerk = np.abs(np.diff(errs))
        sign = np.sign(errs)
        sign[sign == 0] = 1
        crossings = int(np.sum(sign[1:] != sign[:-1]))
    else:
        jerk = np.array([0.0])
        crossings = 0

    return {
        "algorithm": rows[0]["algorithm"],
        "threshold": rows[0]["threshold"],
        "frames": len(rows),
        "detection_rate": round(float(det.mean()), 4),
        "inferred_rate": round(float(np.mean([r["inferred"] for r in rows])), 4),
        "confidence_mean": round(float(conf.mean()), 4),
        "processing_ms_mean": round(float(total.mean()), 2),
        "processing_ms_p95": round(float(np.percentile(total, 95)), 2),
        "detect_ms_mean": round(float(detect.mean()), 2),
        "theoretical_fps": round(1000.0 / max(1e-6, float(total.mean())), 1),
        "mean_abs_error_px": round(float(np.mean(np.abs(errs))), 2) if errs.size else 0.0,
        "error_std_px": round(float(np.std(errs)), 2) if errs.size else 0.0,
        "mean_frame_to_frame_jump_px": round(float(jerk.mean()), 2),
        "zero_crossings": crossings,
    }


# ----------------------------------------------------------------------
def write_markdown(summaries, path):
    cols = ["algorithm", "threshold", "detection_rate", "confidence_mean",
            "processing_ms_mean", "theoretical_fps", "mean_abs_error_px",
            "error_std_px", "mean_frame_to_frame_jump_px"]
    head = {"algorithm": "Algorithm", "threshold": "Threshold",
            "detection_rate": "Detect. rate", "confidence_mean": "Confidence",
            "processing_ms_mean": "Proc. (ms)", "theoretical_fps": "FPS",
            "mean_abs_error_px": "Mean |e| (px)", "error_std_px": "Std e (px)",
            "mean_frame_to_frame_jump_px": "Frame-to-frame jump (px)"}
    with open(path, "w") as fh:
        fh.write("| " + " | ".join(head[c] for c in cols) + " |\n")
        fh.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for s in summaries:
            fh.write("| " + " | ".join(str(s[c]) for c in cols) + " |\n")


def make_plots(all_rows, summaries, out_dir):
    if not HAVE_PLT:
        print("[benchmark] matplotlib not installed, skipping plots")
        return
    os.makedirs(out_dir, exist_ok=True)

    # 1. error traces
    plt.figure(figsize=(10, 4))
    for key, rows in all_rows.items():
        xs = [r["frame"] for r in rows if r["error_px"] != ""]
        ys = [r["error_px"] for r in rows if r["error_px"] != ""]
        plt.plot(xs, ys, linewidth=1.1, label=key)
    plt.axhline(0, color="k", linewidth=0.6)
    plt.xlabel("frame")
    plt.ylabel("lateral error (px)")
    plt.title("Detected lane error over the same footage")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_traces.png"), dpi=150)
    plt.close()

    # 2. processing time distribution
    plt.figure(figsize=(8, 4))
    labels, data = [], []
    for key, rows in all_rows.items():
        labels.append(key)
        data.append([r["total_ms"] for r in rows])
    try:                                  # matplotlib >= 3.9
        plt.boxplot(data, tick_labels=labels, showfliers=False)
    except TypeError:                     # older matplotlib
        plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("frame processing time (ms)")
    plt.xticks(rotation=20, fontsize=8)
    plt.title("Per-frame processing time")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "processing_time.png"), dpi=150)
    plt.close()

    # 3. fps bars
    plt.figure(figsize=(8, 4))
    names = ["%s/%s" % (s["algorithm"], s["threshold"]) for s in summaries]
    vals = [s["theoretical_fps"] for s in summaries]
    plt.bar(names, vals)
    plt.axhline(15, color="r", linestyle="--", label="H4 target: 15 FPS")
    plt.ylabel("FPS (processing only)")
    plt.xticks(rotation=20, fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fps.png"), dpi=150)
    plt.close()
    print("[benchmark] plots written to %s" % out_dir)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Offline algorithm comparison")
    ap.add_argument("--video", help="recorded clip to replay")
    ap.add_argument("--images", help="folder of still test images")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="generate N synthetic frames instead")
    ap.add_argument("--noise", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--thresholds", action="store_true",
                    help="sweep fixed/otsu/adaptive as well (hypothesis H3)")
    ap.add_argument("--sources", action="store_true",
                    help="also sweep the histogram source (edges/binary/combined)")
    ap.add_argument("--config")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    frames = load_frames(args, cfg)
    print("[benchmark] %d frames loaded" % len(frames))

    modes = ["fixed", "otsu", "adaptive"] if args.thresholds else [cfg.vision.threshold_mode]
    sources = ["edges", "binary", "combined"] if args.sources else [cfg.vision.hist_source]
    algorithms = ["canny_hough", "histogram"]
    if args.thresholds and cfg.vision.hist_source == "edges" and not args.sources:
        print("[benchmark] warning: hist_source is 'edges', so the binary "
              "threshold does not\n              affect either detector and the "
              "sweep will show no difference.\n              Use "
              "--sources, or set hist_source to 'combined'.")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, "benchmark_%s" % stamp)
    os.makedirs(out_dir, exist_ok=True)

    all_rows, summaries, flat = {}, [], []
    for algo in algorithms:
      for source in sources:
        for mode in modes:
            rows = run_configuration(frames, cfg, algo, mode, source)
            key = "%s/%s" % (algo, mode)
            if len(sources) > 1:
                key += "/%s" % source
            all_rows[key] = rows
            flat.extend(rows)
            s = summarise(rows)
            s["hist_source"] = source
            summaries.append(s)
            print("  %-30s detect %.3f  %.2f ms  %.1f fps  mean|e| %.1f px"
                  % (key, s["detection_rate"], s["processing_ms_mean"],
                     s["theoretical_fps"], s["mean_abs_error_px"]))

    with open(os.path.join(out_dir, "per_frame.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)

    with open(os.path.join(out_dir, "comparison.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    write_markdown(summaries, os.path.join(out_dir, "comparison.md"))
    with open(os.path.join(out_dir, "comparison.json"), "w") as fh:
        json.dump(summaries, fh, indent=2)
    make_plots(all_rows, summaries, os.path.join(out_dir, "plots"))
    print("\n[benchmark] results in %s" % out_dir)


if __name__ == "__main__":
    main()
