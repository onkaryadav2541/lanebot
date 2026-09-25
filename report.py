#!/usr/bin/env python3
"""
report.py -- Turns recorded runs into thesis figures and tables.

The live dashboard is for driving. This is for the document: it reads
frames.csv from one or more runs and produces vector-quality figures plus
ready-to-paste tables.

Usage
-----
Single run:
    python3 report.py --run runs/histogram_pid_20260904_141233

Compare two or more runs on the same axes (the figures examiners like most):
    python3 report.py --run runs/histogram_pid_... --run runs/histogram_bangbang_... \
                      --labels "PID" "Bang-bang" --name pid_vs_bangbang

Output goes to <run>/figures/ (or results/report_<name>/ when comparing) as
both PNG (300 dpi, for Word) and PDF (vector, for LaTeX), plus:

    metrics_table.md    markdown table for Word
    metrics_table.tex   booktabs table for LaTeX
    figure_captions.md  a caption for every figure, ready to adapt

Figures produced
----------------
     1  lateral error over time, with the deadzone band marked
     2  error distribution (histogram + fitted normal)
     3  cumulative distribution of absolute error
     4  PID term contributions over time (stacked)
     5  controller characteristic: steering command vs error
     6  frame rate over time and its distribution
     7  end-to-end latency distribution with the 80/100 ms thresholds
     8  per-stage timing breakdown (stacked bar + share pie)
     9  detection rate and confidence over time
    10  motor duty cycles over time
    11  detected lane boundaries and lane centre over time
    12  power spectrum of the error signal (oscillation frequency)
    13  phase portrait: error against rate of change of error
    14  latency against frame index with a rolling mean
"""

import argparse
import csv
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# A neutral, print-friendly style. Avoid the matplotlib defaults, which are
# instantly recognisable and look unconsidered in a printed thesis.
plt.rcParams.update({
    "figure.figsize": (7.0, 3.6),
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.3,
})

PALETTE = ["#1f4e79", "#c0504d", "#4f9153", "#8064a2", "#e08214", "#31859c"]
CAPTIONS = {}


# ----------------------------------------------------------------------
def load_run(path):
    """Reads frames.csv (+ summary.json, config.json) from a run folder."""
    if os.path.isdir(path):
        csv_path = os.path.join(path, "frames.csv")
    else:
        csv_path = path
        path = os.path.dirname(path)
    if not os.path.exists(csv_path):
        raise SystemExit("No frames.csv in %s" % path)

    cols = {}
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for name in reader.fieldnames:
            cols[name] = []
        for row in reader:
            for name in reader.fieldnames:
                cols[name].append(row[name])

    def num(name):
        out = []
        for v in cols.get(name, []):
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(np.nan)
        return np.asarray(out, dtype=float)

    data = {name: num(name) for name in cols
            if name not in ("algorithm", "controller", "command")}
    data["command"] = np.asarray(cols.get("command", []))

    # Keep only the frames in which the vehicle was driving. Frames logged
    # before the motors were armed describe a stationary vehicle, and
    # including them makes a short drive look like a long, near-perfect one.
    if "armed" in data and np.any(np.isfinite(data["armed"])):
        keep = data["armed"] > 0
        if keep.sum() >= 10:
            n_total = keep.size
            for k, v in list(data.items()):
                if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape:
                    data[k] = v[keep]
            data["_driven_note"] = ("%d of %d logged frames were driven"
                                    % (int(keep.sum()), n_total))
            print("  using %s" % data["_driven_note"])
    data["algorithm"] = cols.get("algorithm", ["?"])[0] if cols.get("algorithm") else "?"
    data["controller"] = cols.get("controller", ["?"])[0] if cols.get("controller") else "?"
    data["_dir"] = path

    summary_path = os.path.join(path, "summary.json")
    data["summary"] = {}
    if os.path.exists(summary_path):
        with open(summary_path) as fh:
            data["summary"] = json.load(fh)
    cfg_path = os.path.join(path, "config.json")
    data["config"] = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            data["config"] = json.load(fh)
    return data


def finite(a):
    return a[np.isfinite(a)]


def save(fig, out_dir, name, caption):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, "%s.%s" % (name, ext)))
    plt.close(fig)
    CAPTIONS[name] = caption
    print("  %s.png / .pdf" % name)


# ======================================================================
# Figures
# ======================================================================
def fig_error_time(runs, labels, out, px_per_cm=None):
    fig, ax = plt.subplots()
    dz = None
    for r, lab, c in zip(runs, labels, PALETTE):
        t, e = r["t_rel"], r["error_px"]
        m = np.isfinite(e)
        ax.plot(t[m], e[m], color=c, label=lab, alpha=0.9)
        dz = dz or r.get("config", {}).get("control", {}).get("deadzone_px")
    if dz:
        ax.axhspan(-dz, dz, color="#4f9153", alpha=0.12,
                   label="control deadzone (%.0f px)" % dz)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("lateral error (px)")
    if px_per_cm:
        sec = ax.secondary_yaxis(
            "right", functions=(lambda v: v / px_per_cm, lambda v: v * px_per_cm))
        sec.set_ylabel("lateral error (cm)")
    ax.set_title("Lateral error over time")
    ax.legend(ncol=2)
    save(fig, out, "01_error_time",
         "Lateral deviation of the detected lane centre from the image centre "
         "over the duration of the run. The shaded band is the control "
         "deadzone, inside which no steering correction is issued.")


def fig_error_distribution(runs, labels, out):
    fig, ax = plt.subplots()
    for r, lab, c in zip(runs, labels, PALETTE):
        e = finite(r["error_px"])
        if e.size == 0:
            continue
        # Step outline rather than solid bars: with two configurations
        # overlaid, filled bars hide each other and the comparison is lost.
        ax.hist(e, bins=45, histtype="stepfilled", alpha=0.30, color=c, density=True)
        ax.hist(e, bins=45, histtype="step", lw=1.6, color=c, density=True,
                label="%s  (median %.1f, IQR %.1f px)"
                      % (lab, np.median(e), np.percentile(e, 75) - np.percentile(e, 25)))
        ax.axvline(np.median(e), color=c, ls=":", lw=1.2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("lateral error (px)")
    ax.set_ylabel("probability density")
    ax.set_title("Distribution of the lateral error")
    ax.legend()
    save(fig, out, "02_error_distribution",
         "Probability density of the lateral error, with the median marked. "
         "The distribution is reported by median and interquartile range "
         "rather than by a normal fit, because a vehicle oscillating about "
         "the lane centre produces a bimodal rather than a normal "
         "distribution.")


def fig_error_cdf(runs, labels, out):
    fig, ax = plt.subplots()
    for r, lab, c in zip(runs, labels, PALETTE):
        e = np.abs(finite(r["error_px"]))
        if e.size == 0:
            continue
        xs = np.sort(e)
        ys = np.arange(1, xs.size + 1) / xs.size
        ax.plot(xs, ys, color=c, label=lab)
        p95 = np.percentile(e, 95)
        ax.plot([p95], [0.95], "o", color=c, ms=4)
        ax.annotate("p95 = %.0f px" % p95, (p95, 0.95),
                    textcoords="offset points", xytext=(6, -10),
                    fontsize=7, color=c)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("absolute lateral error (px)")
    ax.set_ylabel("fraction of frames")
    ax.set_title("Cumulative distribution of the absolute lateral error")
    ax.legend()
    save(fig, out, "03_error_cdf",
         "Empirical cumulative distribution of the absolute lateral error. "
         "The 95th percentile is a stricter accuracy statement than the mean, "
         "since it bounds all but the worst 5 % of frames.")


def fig_pid_terms(run, out, label=""):
    if not np.any(np.isfinite(run.get("p_term", np.array([np.nan])))):
        return
    fig, ax = plt.subplots()
    t = run["t_rel"]
    for key, name, c in (("p_term", "P", PALETTE[2]),
                         ("i_term", "I", PALETTE[0]),
                         ("d_term", "D", PALETTE[1])):
        ax.plot(t, run[key], color=c, label=name, lw=1.1)
    ax.plot(t, run["steer"], color="k", lw=1.4, label="steer (sum, clipped)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalised contribution")
    ax.set_title("PID term contributions %s" % label)
    ax.legend(ncol=4)
    save(fig, out, "04_pid_terms",
         "Individual contributions of the proportional, integral and "
         "derivative terms to the steering command. The proportional term "
         "dominates the transient response, the integral term removes the "
         "steady-state offset in sustained curves, and the derivative term "
         "opposes rapid changes and so damps overshoot.")


def fig_controller_characteristic(runs, labels, out):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for idx, (r, lab, c) in enumerate(zip(runs, labels, PALETTE)):
        e, st = r["error_px"], r["steer"]
        m = np.isfinite(e) & np.isfinite(st)
        e, st = e[m], st[m]
        if e.size == 0:
            continue
        # A discrete controller puts thousands of points on three exact
        # values, so they stack invisibly. A little vertical jitter and a
        # larger marker make that structure visible without misrepresenting
        # the data; the jitter is stated in the caption.
        discrete = np.unique(np.round(st, 3)).size <= 5
        y = st + (np.random.default_rng(0).normal(0, 0.012, st.size)
                  if discrete else 0.0)
        ax.scatter(e, y, s=22 if discrete else 7,
                   alpha=0.45 if discrete else 0.30, color=c,
                   marker="s" if discrete else "o", edgecolors="none",
                   label="%s%s" % (lab, " (jittered)" if discrete else ""),
                   zorder=3 - idx)
        # binned mean shows the underlying transfer function
        if not discrete and e.size > 40:
            bins = np.linspace(e.min(), e.max(), 18)
            which = np.digitize(e, bins)
            xs, ys = [], []
            for b in range(1, len(bins)):
                sel = which == b
                if sel.sum() > 3:
                    xs.append(e[sel].mean())
                    ys.append(st[sel].mean())
            ax.plot(xs, ys, color=c, lw=2.2, zorder=5)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("lateral error (px)")
    ax.set_ylabel("steering command")
    ax.set_ylim(-1.25, 1.25)
    ax.set_title("Controller characteristic: commanded steering against measured error")
    ax.legend(loc="upper left")
    save(fig, out, "05_controller_characteristic",
         "Steering command against measured lateral error for every frame. "
         "The bang-bang controller occupies three discrete levels; the PID "
         "controller produces a continuous, approximately proportional "
         "response, shown by the binned mean. Discrete levels are drawn with "
         "slight vertical jitter so that overlapping points remain visible.")


def fig_fps(runs, labels, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0),
                                 gridspec_kw={"width_ratios": [2, 1]})
    for r, lab, c in zip(runs, labels, PALETTE):
        f = r["fps"]
        m = np.isfinite(f)
        a1.plot(r["t_rel"][m], f[m], color=c, lw=0.9, alpha=0.75, label=lab)
        if m.sum() > 15:
            k = np.ones(15) / 15
            a1.plot(r["t_rel"][m][14:], np.convolve(f[m], k, "valid"),
                    color=c, lw=1.6)
        a2.hist(finite(f), bins=25, orientation="horizontal",
                color=c, alpha=0.6)
    a1.axhline(15, color="#c0504d", ls="--", lw=1.0, label="H4 target 15 FPS")
    a1.set_xlabel("time (s)")
    a1.set_ylabel("frame rate (FPS)")
    a1.set_title("Frame rate over time")
    a1.legend(ncol=2)
    a2.axhline(15, color="#c0504d", ls="--", lw=1.0)
    a2.set_xlabel("frames")
    a2.set_title("Distribution")
    save(fig, out, "06_frame_rate",
         "Instantaneous frame rate over the run (thin line) with a 15-frame "
         "moving average (thick line), and the corresponding distribution. "
         "The dashed line marks the 15 FPS real-time target of hypothesis H4.")


def fig_latency(runs, labels, out):
    fig, ax = plt.subplots()
    for r, lab, c in zip(runs, labels, PALETTE):
        lat = finite(r["latency_ms"])
        if lat.size == 0:
            continue
        ax.hist(lat, bins=40, alpha=0.6, color=c,
                label="%s (mean %.1f ms, p95 %.1f ms)"
                      % (lab, lat.mean(), np.percentile(lat, 95)))
    ax.axvline(80, color="#4f9153", ls="--", label="H2 target 80 ms")
    ax.axvline(100, color="#c0504d", ls="--", label="H2 limit 100 ms")
    ax.set_xlabel("end-to-end latency, capture to motor command (ms)")
    ax.set_ylabel("frames")
    ax.set_title("End-to-end latency distribution")
    ax.legend()
    save(fig, out, "07_latency",
         "Distribution of the measured end-to-end latency from image capture "
         "to the issued motor command. The two reference lines are the "
         "thresholds stated in hypothesis H2.")


def fig_stage_timing(run, out, label=""):
    stages = [("t_grayscale", "grayscale"), ("t_blur", "Gaussian blur"),
              ("t_threshold", "thresholding"), ("t_canny", "Canny"),
              ("t_roi", "ROI mask"), ("t_detect", "lane detection"),
              ("t_control", "PID control")]
    display = [("t_render", "telemetry display")]

    def collect(items):
        names, means, sds = [], [], []
        for key, name in items:
            v = finite(run.get(key, np.array([])))
            if v.size:
                names.append(name)
                means.append(float(v.mean()))
                sds.append(float(v.std()))
        return names, means, sds

    names, means, sds = collect(stages)
    dnames, dmeans, dsds = collect(display)
    if not means:
        return

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.8),
                                 gridspec_kw={"width_ratios": [3, 2]})

    # Horizontal bars, one per stage, rather than a single stacked bar. A
    # stacked bar is unreadable here because the display stage is an order of
    # magnitude larger than the algorithmic ones and compresses them to
    # slivers. The display is shown separately for the same reason: it is not
    # part of the autonomous pipeline.
    y = np.arange(len(names))
    a1.barh(y, means, xerr=sds, color=PALETTE[0], height=0.62,
            error_kw={"ecolor": "#888888", "elinewidth": 1.0, "capsize": 3})
    a1.set_yticks(y)
    a1.set_yticklabels(names)
    a1.invert_yaxis()
    a1.set_xlabel("mean time per frame (ms)")
    a1.set_title("Pipeline stages (total %.2f ms)" % sum(means))
    for i, (v, sd) in enumerate(zip(means, sds)):
        a1.text(v + sd + max(means) * 0.035, i, "%.2f" % v,
                va="center", fontsize=8)
    a1.set_xlim(0, (max(m + s for m, s in zip(means, sds))) * 1.30)
    a1.grid(axis="y", visible=False)

    shares = np.array(means) / sum(means) * 100
    order = np.argsort(shares)[::-1]
    a2.barh(np.arange(len(names)), shares[order], color=PALETTE[2], height=0.62)
    a2.set_yticks(np.arange(len(names)))
    a2.set_yticklabels([names[i] for i in order])
    a2.invert_yaxis()
    a2.set_xlabel("share of pipeline time (%)")
    a2.set_title("Relative cost")
    for i, v in enumerate(shares[order]):
        a2.text(v + 1.5, i, "%.0f%%" % v, va="center", fontsize=8)
    a2.set_xlim(0, max(shares) * 1.3)
    a2.grid(axis="y", visible=False)

    note = "pipeline %.2f ms per frame" % sum(means)
    if dmeans:
        note += "   ·   telemetry display %.1f ms (development only, " \
                "not part of autonomous operation)" % dmeans[0]
    fig.suptitle("Processing time budget %s" % label, fontsize=10)
    fig.text(0.5, -0.02, note, ha="center", fontsize=8, color="#555555")
    fig.tight_layout()
    save(fig, out, "08_stage_timing",
         "Mean execution time of each pipeline stage on the target hardware, "
         "with standard deviation. The telemetry display is excluded from the "
         "bars and reported separately, since it runs only during development "
         "and dominates the budget by an order of magnitude.")


def fig_detection(runs, labels, out):
    fig, ax = plt.subplots()
    notes = []
    for r, lab, c in zip(runs, labels, PALETTE):
        t, conf = r["t_rel"], r["confidence"]
        ax.plot(t, conf, color=c, lw=0.7, alpha=0.35)
        if conf.size > 25:
            k = np.ones(25) / 25
            ax.plot(t[24:], np.convolve(conf, k, "valid"), color=c, lw=1.8,
                    label="%s (mean %.2f)" % (lab, np.nanmean(conf)))
        rate = 100 * float(np.nanmean(r["detected"]))
        notes.append("%s: %.1f %% of frames detected" % (lab, rate))
        # mark any frame where detection was lost, since those are the
        # moments that matter and a rolling average hides them
        lost = t[r["detected"] == 0]
        if lost.size:
            ax.plot(lost, np.full(lost.size, 0.02), "|", color=c, ms=10)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("detection confidence")
    ax.set_ylim(0, 1.02)
    ax.set_title("Lane detection reliability")
    ax.legend(loc="lower right")
    ax.text(0.01, 0.97, "\n".join(notes), transform=ax.transAxes, fontsize=8,
            va="top", color="#444444")
    save(fig, out, "09_detection",
         "Per-frame detection confidence with a 25-frame moving average. "
         "Tick marks along the bottom axis mark frames in which no lane was "
         "detected; the overall detection rate is stated inset.")


def fig_motor_duty(run, out, label=""):
    l, r_ = run.get("left_duty"), run.get("right_duty")
    if l is None or not np.any(np.isfinite(l)):
        return
    fig, ax = plt.subplots()
    ax.plot(run["t_rel"], l, color=PALETTE[0], label="left channel")
    ax.plot(run["t_rel"], r_, color=PALETTE[1], label="right channel")
    ax.fill_between(run["t_rel"], l, r_, color="#999999", alpha=0.18,
                    label="differential")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("PWM duty cycle (%)")
    ax.set_title("Motor duty cycles %s" % label)
    ax.legend(ncol=3)
    save(fig, out, "10_motor_duty",
         "Commanded PWM duty cycle of the left and right motor channels. The "
         "shaded area is the differential between the two channels, which is "
         "what produces the yaw rate of the vehicle.")


def fig_lane_positions(run, out, label=""):
    fig, ax = plt.subplots()
    t = run["t_rel"]
    ax.plot(t, run.get("left_x", np.full_like(t, np.nan)), ".", ms=2,
            color=PALETTE[0], label="left boundary")
    ax.plot(t, run.get("right_x", np.full_like(t, np.nan)), ".", ms=2,
            color=PALETTE[1], label="right boundary")
    ax.plot(t, run["lane_center"], color=PALETTE[2], lw=1.2, label="lane centre")
    width = run.get("config", {}).get("camera", {}).get("width", 320)
    ax.axhline(width / 2, color="k", ls="--", lw=0.8, label="image centre")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("image column (px)")
    ax.set_title("Detected lane boundaries %s" % label)
    ax.legend(ncol=2)
    save(fig, out, "11_lane_positions",
         "Detected left and right lane boundary positions and the resulting "
         "lane centre, in image coordinates. Gaps in the boundary traces are "
         "frames where only one boundary was visible and the other was "
         "inferred from the estimated lane width.")


def fig_spectrum(runs, labels, out):
    fig, ax = plt.subplots()
    any_data = False
    for r, lab, c in zip(runs, labels, PALETTE):
        e, t = r["error_px"], r["t_rel"]
        m = np.isfinite(e)
        e, t = e[m], t[m]
        if e.size < 64:
            continue
        dt = float(np.median(np.diff(t))) if t.size > 1 else 1 / 15.0

        # Welch-style averaging: split the record into overlapping segments
        # and average their spectra. A single FFT of a short noisy record is
        # dominated by random variation between neighbouring bins, which is
        # what makes the raw periodogram unreadable.
        seg = min(256, e.size // 3)
        if seg < 32:
            seg = e.size
        step = max(1, seg // 2)
        win = np.hanning(seg)
        acc, count = None, 0
        for start in range(0, e.size - seg + 1, step):
            chunk = e[start:start + seg]
            chunk = chunk - chunk.mean()
            spec = np.abs(np.fft.rfft(chunk * win)) / seg
            acc = spec if acc is None else acc + spec
            count += 1
        if count == 0:
            continue
        spec = acc / count
        freq = np.fft.rfftfreq(seg, dt)

        ax.semilogy(freq[1:], spec[1:], color=c, lw=1.6, label=lab)
        peak_i = int(np.argmax(spec[1:])) + 1
        ax.plot(freq[peak_i], spec[peak_i], "o", color=c, ms=6)
        ax.annotate("%.2f Hz" % freq[peak_i], (freq[peak_i], spec[peak_i]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, color=c)
        any_data = True
    if not any_data:
        plt.close(fig)
        return
    ax.set_xlim(0, 5)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("amplitude (px)")
    ax.set_title("Averaged power spectrum of the lateral error")
    ax.legend()
    save(fig, out, "12_error_spectrum",
         "Amplitude spectrum of the lateral error, averaged over overlapping "
         "Hanning-windowed segments to suppress the bin-to-bin variance of a "
         "single periodogram. The marked peak is the dominant oscillation "
         "frequency: a lower peak amplitude corresponds to smoother steering, "
         "which is the quantitative form of hypothesis H1.")


def fig_phase_portrait(runs, labels, out):
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.9), squeeze=False)
    for ax, r, lab, c in zip(axes[0], runs, labels, PALETTE):
        e, t = r["error_px"], r["t_rel"]
        m = np.isfinite(e)
        e, t = e[m], t[m]
        if e.size < 10:
            continue
        # The derivative of a pixel-quantised signal is extremely noisy, and
        # plotting it raw produces an unreadable tangle. Smoothing the error
        # before differentiating recovers the trajectory shape that the plot
        # is meant to show.
        k = np.ones(9) / 9
        es = np.convolve(e, k, "same")
        de = np.gradient(es, t)

        # One panel per configuration, and a density image rather than a line,
        # so that overlapping trajectories accumulate instead of hiding each
        # other.
        ax.hexbin(es, de, gridsize=34, cmap="Blues", mincnt=1, linewidths=0)
        ax.plot(es, de, color=c, lw=0.5, alpha=0.25)
        ax.axhline(0, color="k", lw=0.7)
        ax.axvline(0, color="k", lw=0.7)
        ax.plot(0, 0, "+", color="#c0504d", ms=12, mew=2)
        ax.set_xlabel("error (px)")
        ax.set_ylabel("rate of change of error (px/s)")
        ax.set_title(lab)
    fig.suptitle("Phase portrait of the closed-loop error", fontsize=10)
    fig.tight_layout()
    save(fig, out, "13_phase_portrait",
         "Phase portrait of the lateral error against its rate of change, one "
         "panel per configuration, shaded by the density of visited states. "
         "The error is smoothed before differentiation because differentiating "
         "a pixel-quantised signal directly yields mostly noise. Trajectories "
         "concentrated near the origin indicate a well-damped controller; a "
         "closed ring indicates a sustained limit cycle.")


def fig_latency_series(runs, labels, out):
    fig, ax = plt.subplots()
    for r, lab, c in zip(runs, labels, PALETTE):
        lat = r["latency_ms"]
        m = np.isfinite(lat)
        ax.plot(r["frame"][m], lat[m], color=c, lw=0.6, alpha=0.45)
        if m.sum() > 25:
            k = np.ones(25) / 25
            ax.plot(r["frame"][m][24:], np.convolve(lat[m], k, "valid"),
                    color=c, lw=1.5, label=lab)
    ax.axhline(80, color="#4f9153", ls="--", lw=1.0, label="80 ms")
    ax.axhline(100, color="#c0504d", ls="--", lw=1.0, label="100 ms")
    ax.set_xlabel("frame index")
    ax.set_ylabel("latency (ms)")
    ax.set_title("End-to-end latency over the run")
    ax.legend(ncol=3)
    save(fig, out, "14_latency_series",
         "End-to-end latency per frame (thin) and its 25-frame moving average "
         "(thick). Periodic excursions typically indicate CPU contention or "
         "thermal throttling of the Raspberry Pi rather than algorithmic cost.")


# ======================================================================
# Tables
# ======================================================================
def compute_metrics(run, px_per_cm=None):
    e = finite(run["error_px"])
    lat = finite(run["latency_ms"])
    fps = finite(run["fps"])
    fps = fps[fps > 0]          # the first frame has no measured loop period
    det = run["detected"]
    steer = finite(run["steer"])

    if e.size > 1:
        sign = np.sign(e)
        sign[sign == 0] = 1
        crossings = int(np.sum(sign[1:] != sign[:-1]))
        duration = float(run["t_rel"][-1] - run["t_rel"][0])
        cross_rate = crossings / max(0.01, duration)
    else:
        crossings, cross_rate = 0, 0.0

    m = {
        "Algorithm": run["algorithm"],
        "Controller": run["controller"],
        "Frames": int(np.isfinite(run["frame"]).sum()),
        "Duration (s)": round(float(run["t_rel"][-1]), 1) if run["t_rel"].size else 0,
        "Detection rate (%)": round(100 * float(np.nanmean(det)), 1) if det.size else 0,
        "Mean |error| (px)": round(float(np.mean(np.abs(e))), 2) if e.size else 0,
        "RMS error (px)": round(float(np.sqrt(np.mean(e ** 2))), 2) if e.size else 0,
        "p95 |error| (px)": round(float(np.percentile(np.abs(e), 95)), 2) if e.size else 0,
        "Max |error| (px)": round(float(np.max(np.abs(e))), 2) if e.size else 0,
        "Error bias (px)": round(float(np.mean(e)), 2) if e.size else 0,
        "Error std (px)": round(float(np.std(e)), 2) if e.size else 0,
        "Centre crossings (1/s)": round(cross_rate, 2),
        "Steer activity": round(float(np.mean(np.abs(np.diff(steer)))), 4)
        if steer.size > 1 else 0,
        "Throughput (FPS)": round(float(run["frame"][-1] / run["t_rel"][-1]), 1)
        if run["t_rel"].size and run["t_rel"][-1] > 0 else 0,
        "Median FPS": round(float(np.median(fps)), 1) if fps.size else 0,
        "Min FPS": round(float(np.min(fps)), 1) if fps.size else 0,
        "Mean latency (ms)": round(float(np.mean(lat)), 1) if lat.size else 0,
        "p95 latency (ms)": round(float(np.percentile(lat, 95)), 1) if lat.size else 0,
    }
    if px_per_cm:
        m["Mean |error| (cm)"] = round(m["Mean |error| (px)"] / px_per_cm, 2)
        m["RMS error (cm)"] = round(m["RMS error (px)"] / px_per_cm, 2)
    return m


def write_tables(metrics, labels, out_dir):
    keys = list(metrics[0].keys())

    md = ["| Metric | " + " | ".join(labels) + " |",
          "|---" * (len(labels) + 1) + "|"]
    for k in keys:
        md.append("| %s | " % k + " | ".join(str(m[k]) for m in metrics) + " |")
    with open(os.path.join(out_dir, "metrics_table.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")

    tex = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{Measured performance metrics.}",
           r"\label{tab:results}",
           r"\begin{tabular}{l" + "r" * len(labels) + "}", r"\hline",
           "Metric & " + " & ".join(labels) + r" \\", r"\hline"]
    for k in keys:
        safe = k.replace("%", r"\%").replace("|", r"$|$")
        tex.append("%s & " % safe + " & ".join(str(m[k]) for m in metrics) + r" \\")
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(out_dir, "metrics_table.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")

    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump({lab: m for lab, m in zip(labels, metrics)}, fh, indent=2)

    print("\n  " + "%-26s" % "Metric" + "".join("%16s" % l for l in labels))
    print("  " + "-" * (26 + 16 * len(labels)))
    for k in keys:
        print("  " + "%-26s" % k + "".join("%16s" % m[k] for m in metrics))


def write_captions(out_dir):
    lines = ["# Figure captions", ""]
    for name in sorted(CAPTIONS):
        pretty = name.split("_", 1)[1].replace("_", " ").title()
        lines += ["## %s (`%s.png`)" % (pretty, name), "", CAPTIONS[name], ""]
    with open(os.path.join(out_dir, "figure_captions.md"), "w") as fh:
        fh.write("\n".join(lines))


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="Generate thesis figures")
    ap.add_argument("--run", action="append", required=True,
                    help="run folder (repeat to compare several runs)")
    ap.add_argument("--labels", nargs="*", help="legend label per run")
    ap.add_argument("--name", default="report",
                    help="output name when comparing several runs")
    ap.add_argument("--px-per-cm", type=float,
                    help="scale from calibrate.py, adds a cm axis and cm metrics")
    ap.add_argument("--out", help="output directory (default: inside the run)")
    args = ap.parse_args()

    runs = [load_run(p) for p in args.run]
    labels = args.labels if args.labels else [
        "%s / %s" % (r["algorithm"], r["controller"]) for r in runs]
    if len(labels) != len(runs):
        raise SystemExit("Give one label per run")

    if args.out:
        out_dir = args.out
    elif len(runs) == 1:
        out_dir = os.path.join(runs[0]["_dir"], "figures")
    else:
        out_dir = os.path.join("results", "report_%s" % args.name)
    os.makedirs(out_dir, exist_ok=True)
    print("Writing figures to %s" % out_dir)

    fig_error_time(runs, labels, out_dir, args.px_per_cm)
    fig_error_distribution(runs, labels, out_dir)
    fig_error_cdf(runs, labels, out_dir)
    fig_pid_terms(runs[0], out_dir, "(%s)" % labels[0])
    fig_controller_characteristic(runs, labels, out_dir)
    fig_fps(runs, labels, out_dir)
    fig_latency(runs, labels, out_dir)
    fig_stage_timing(runs[0], out_dir, "(%s)" % labels[0])
    fig_detection(runs, labels, out_dir)
    fig_motor_duty(runs[0], out_dir, "(%s)" % labels[0])
    fig_lane_positions(runs[0], out_dir, "(%s)" % labels[0])
    fig_spectrum(runs, labels, out_dir)
    fig_phase_portrait(runs, labels, out_dir)
    fig_latency_series(runs, labels, out_dir)

    metrics = [compute_metrics(r, args.px_per_cm) for r in runs]
    write_tables(metrics, labels, out_dir)
    write_captions(out_dir)
    print("\nTables: metrics_table.md, metrics_table.tex, metrics.json")
    print("Captions: figure_captions.md\n")


if __name__ == "__main__":
    main()
