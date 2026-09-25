#!/usr/bin/env python3
"""
make_track.py -- Printable test material.

Generates exact-scale PDFs you can print at home on a normal A4 printer.

    python3 make_track.py --bench        # 9 bench-test sheets  <- start here
    python3 make_track.py --calibration  # ruler + checkerboard sheet
    python3 make_track.py --tiles 6      # floor tiles for actual driving
    python3 make_track.py --all

    python3 make_track.py --bench --lane-width 160 --line-width 12
    python3 make_track.py --bench --paper a3

PRINTING, IMPORTANT
-------------------
In the print dialogue set scaling to "Actual size" / "100 %", NOT "Fit to
page". Every sheet carries a 100 mm reference bar: measure it with a ruler
after printing. If it is not 100 mm your printer scaled the page and every
dimension in your results chapter will be wrong.

WHAT TO USE WHEN
----------------
Bench sheets    Car stationary, camera pointed at the sheet. Validates the
                perception pipeline and the generated steering commands. This
                is the software-in-the-loop validation your research paper
                already used, and it is repeatable, which a floor test is not.

Calibration     Gives you pixels-per-centimetre, so lateral deviation can be
                reported in cm instead of pixels. Examiners ask for this.

Floor tiles     For driving. Honestly, black masking tape or electrical tape
                on a light floor is cheaper, tougher and easier to lay out
                than 20 taped-together printouts -- the tiles are here for
                when you want a controlled, documented track for photographs.
"""

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, Polygon

MM = 1.0 / 25.4          # mm -> inch

PAPER = {
    "a4": (297.0, 210.0),        # landscape
    "a4p": (210.0, 297.0),       # portrait
    "a3": (420.0, 297.0),
    "letter": (279.4, 215.9),
}


# ----------------------------------------------------------------------
def new_page(pdf, w_mm, h_mm, title, subtitle="", margin=8.0):
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_mm)
    ax.set_ylim(0, h_mm)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), w_mm, h_mm, facecolor="white",
                           edgecolor="none", zorder=0))
    # header
    ax.text(margin, h_mm - margin + 1, title, fontsize=7, color="0.45",
            va="top", ha="left", zorder=5)
    if subtitle:
        ax.text(w_mm - margin, h_mm - margin + 1, subtitle, fontsize=7,
                color="0.45", va="top", ha="right", zorder=5)
    return fig, ax


def scale_bar(ax, x, y, length=100.0):
    """A printed reference so you can verify the printer did not rescale."""
    ax.plot([x, x + length], [y, y], color="0.35", lw=1.0, zorder=5)
    for xx in (x, x + length):
        ax.plot([xx, xx], [y - 2, y + 2], color="0.35", lw=1.0, zorder=5)
    for i in range(1, 10):
        ax.plot([x + i * length / 10] * 2, [y - 1, y + 1],
                color="0.55", lw=0.6, zorder=5)
    ax.text(x + length / 2, y + 3, "%d mm reference - measure this after printing"
            % length, fontsize=6, color="0.35", ha="center", zorder=5)


def edge_marks(ax, w, h, positions):
    """Alignment ticks where the lane meets a page edge, for taping tiles."""
    for (x, y, side) in positions:
        if side == "left":
            ax.plot([0, 6], [y, y], color="0.6", lw=0.6, zorder=5)
        elif side == "right":
            ax.plot([w - 6, w], [y, y], color="0.6", lw=0.6, zorder=5)
        elif side == "bottom":
            ax.plot([x, x], [0, 6], color="0.6", lw=0.6, zorder=5)
        else:
            ax.plot([x, x], [h - 6, h], color="0.6", lw=0.6, zorder=5)


def draw_lane_band(ax, xs, ys, lane_width, line_width, dashed=False,
                   dash_len=40.0, gap_len=30.0):
    """
    Draws two lane boundary lines offset from the centreline (xs, ys).

    The offset is taken along the local normal, so curves keep a constant
    lane width instead of pinching on the inside of the bend.
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    norm = np.hypot(dx, dy)
    norm[norm == 0] = 1e-9
    nx, ny = -dy / norm, dx / norm
    half = lane_width / 2.0

    seg_len = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])

    for sign in (-1, 1):
        cx = xs + sign * half * nx
        cy = ys + sign * half * ny
        if not dashed:
            ax.plot(cx, cy, color="black", lw=0, zorder=3)
            _thick_line(ax, cx, cy, line_width)
        else:
            period = dash_len + gap_len
            on = (seg_len % period) < dash_len
            start = None
            for i, flag in enumerate(on):
                if flag and start is None:
                    start = i
                if (not flag or i == len(on) - 1) and start is not None:
                    if i - start > 1:
                        _thick_line(ax, cx[start:i], cy[start:i], line_width)
                    start = None


def _thick_line(ax, cx, cy, width):
    """Draws a line of exact physical width as a filled polygon."""
    cx = np.asarray(cx, float)
    cy = np.asarray(cy, float)
    if cx.size < 2:
        return
    dx = np.gradient(cx)
    dy = np.gradient(cy)
    n = np.hypot(dx, dy)
    n[n == 0] = 1e-9
    nx, ny = -dy / n, dx / n
    h = width / 2.0
    upper = np.column_stack([cx + h * nx, cy + h * ny])
    lower = np.column_stack([cx - h * nx, cy - h * ny])[::-1]
    ax.add_patch(Polygon(np.vstack([upper, lower]), closed=True,
                         facecolor="black", edgecolor="none", zorder=3))


# ======================================================================
# Bench sheets: the car stands still, the camera looks at the sheet
# ======================================================================
def bench_sheets(pdf, paper, lane_width, line_width):
    w, h = PAPER[paper]
    cx, cy = w / 2.0, h / 2.0

    def straight(offset=0.0, label=""):
        ys = np.linspace(10, h - 20, 300)
        xs = np.full_like(ys, cx + offset)
        return xs, ys, label

    sheets = []

    # the largest offset that still keeps both lines on the paper
    max_off = (w - lane_width) / 2.0 - line_width / 2.0 - 6.0
    off_mm = float(np.clip(np.floor(max_off / 5.0) * 5.0, 0.0, 60.0))

    variants = [(0.0, "centred")]
    if off_mm >= 10.0:
        variants += [(-off_mm, "offset %d mm left" % off_mm),
                     (off_mm, "offset %d mm right" % off_mm)]
    for off, name in variants:
        sheets.append(("Straight lane, %s" % name, straight(off)[:2], False,
                       "the lane centre is %+d mm from the sheet centre" % int(off),
                       lane_width, line_width))

    # gentle and sharp curves
    for radius, name in ((600.0, "gentle curve, R = 600 mm"),
                         (300.0, "sharp curve, R = 300 mm")):
        t = np.linspace(0, min(1.0, (h - 30) / radius), 300)
        xs = cx + radius * (1 - np.cos(t))
        ys = 10 + radius * np.sin(t)
        keep = (ys <= h - 15) & (xs > 10) & (xs < w - 10)
        sheets.append((name, (xs[keep], ys[keep]), False,
                       "curve to the right", lane_width, line_width))
        sheets.append((name + ", mirrored", (2 * cx - xs[keep], ys[keep]), False,
                       "curve to the left", lane_width, line_width))

    # dashed markings
    ys = np.linspace(10, h - 20, 400)
    xs = np.full_like(ys, cx)
    sheets.append(("Dashed lane markings", (xs, ys), True,
                   "tests robustness to broken lines", lane_width, line_width))

    # narrow and wide lane
    sheets.append(("Narrow lane (%d mm)" % int(lane_width * 0.7),
                   straight(0)[:2], False, "tests lane-width adaptation",
                   lane_width * 0.7, line_width))
    wide = min(lane_width * 1.3, w - line_width - 20.0)
    sheets.append(("Wide lane (%d mm)" % int(wide),
                   straight(0)[:2], False, "tests lane-width adaptation",
                   wide, line_width))

    # single visible boundary: the recovery path in the detector
    ys = np.linspace(10, h - 20, 300)
    sheets.append(("Single boundary (right line only)", None, False,
                   "the missing boundary must be inferred",
                   lane_width, line_width))

    for i, (title, path, dashed, note, lw_lane, lw_line) in enumerate(sheets, 1):
        fig, ax = new_page(pdf, w, h,
                           "Bench test sheet %d/%d - %s" % (i, len(sheets), title),
                           "lane %d mm, line %d mm" % (lw_lane, lw_line))
        if path is None:
            _thick_line(ax, np.full_like(ys, cx + lw_lane / 2), ys, lw_line)
        else:
            draw_lane_band(ax, path[0], path[1], lw_lane, lw_line, dashed=dashed)
        ax.text(w / 2, 6, note, fontsize=7, color="0.45", ha="center")
        scale_bar(ax, w - 118, h - 16)
        pdf.savefig(fig)
        plt.close(fig)
    return len(sheets)


# ======================================================================
# Calibration sheet
# ======================================================================
def calibration_sheet(pdf, paper, lane_width, line_width):
    w, h = PAPER[paper]
    fig, ax = new_page(pdf, w, h, "Calibration sheet",
                       "print at 100 %, then verify with a ruler")

    # 200 mm ruler with 10 mm ticks, across the top
    y = h - 22
    x0 = (w - 200) / 2
    ax.plot([x0, x0 + 200], [y, y], color="black", lw=1.2)
    for i in range(0, 21):
        xx = x0 + i * 10
        long = (i % 5 == 0)
        ax.plot([xx, xx], [y, y + (7 if long else 4)], color="black",
                lw=1.2 if long else 0.7)
        if long:
            ax.text(xx, y + 9, "%d" % (i * 10), fontsize=6, ha="center")
    ax.text(w / 2, y - 6, "200 mm ruler (10 mm divisions)", fontsize=7,
            color="0.4", ha="center")

    # checkerboard, 20 mm squares -- also usable for lens distortion work
    cols, rows, sq = 9, 3, 20.0
    bx = (w - cols * sq) / 2
    by = h - 115.0
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                ax.add_patch(Rectangle((bx + c * sq, by + r * sq), sq, sq,
                                       facecolor="black", edgecolor="none"))
    ax.add_patch(Rectangle((bx, by), cols * sq, rows * sq, fill=False,
                           edgecolor="0.6", lw=0.5))
    ax.text(w / 2, by - 7, "%dx%d checkerboard, %d mm squares"
            % (cols, rows, int(sq)), fontsize=7, color="0.4", ha="center")

    # a lane of known width for the pixels-per-mm measurement
    y_lo, y_hi = 30.0, by - 22.0
    ys = np.linspace(y_lo, y_hi, 100)
    xs = np.full_like(ys, w / 2)
    draw_lane_band(ax, xs, ys, lane_width, line_width)
    y_mid = 0.5 * (y_lo + y_hi)
    ax.annotate("", xy=(w / 2 - lane_width / 2, y_mid),
                xytext=(w / 2 + lane_width / 2, y_mid),
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=0.8))
    ax.text(w / 2, y_mid + 4,
            "known lane width = %d mm (centre to centre)" % int(lane_width),
            fontsize=7, color="0.35", ha="center")

    scale_bar(ax, 12, 16)
    ax.text(w - 12, 16, "calibrate.py --mode scale", fontsize=7,
            color="0.45", ha="right")
    pdf.savefig(fig)
    plt.close(fig)


# ======================================================================
# Floor tiles
# ======================================================================
def floor_tiles(pdf, paper, lane_width, line_width, count):
    w, h = PAPER[paper]
    usable = h - 2 * line_width - 24.0
    if lane_width > usable:
        print("  note: a %d mm lane does not fit across %s (max %d mm)."
              % (lane_width, paper.upper(), usable))
        print("        Tiles are drawn at %d mm. For a wider driving lane use"
              % usable)
        print("        --paper a3, or lay the lane out with black tape instead.")
        lane_width = usable
    made = 0
    for i in range(count):
        kind = ["straight", "straight", "curve_right", "straight",
                "curve_left", "straight"][i % 6]
        fig, ax = new_page(pdf, w, h, "Floor tile %d - %s" % (i + 1, kind),
                           "butt the edges together and tape underneath")
        if kind == "straight":
            xs = np.linspace(0, w, 200)
            ys = np.full_like(xs, h / 2)
            edge_marks(ax, w, h, [(0, h / 2, "left"), (0, h / 2, "right")])
        else:
            sign = 1.0 if kind == "curve_right" else -1.0
            # lateral shift across the tile, entering and leaving level so
            # that any tile can follow any other tile
            xs = np.linspace(0, w, 300)
            shift = (h / 2 - lane_width / 2 - 15)
            ys = h / 2 + sign * shift * 0.5 * (1 - np.cos(np.pi * xs / w))
            edge_marks(ax, w, h, [(0, ys[0], "left"), (0, ys[-1], "right")])
        draw_lane_band(ax, xs, ys, lane_width, line_width)
        scale_bar(ax, w - 118, 12)
        pdf.savefig(fig)
        plt.close(fig)
        made += 1
    return made


# ======================================================================
def instructions_page(pdf, paper, lane_width, line_width, camera_note):
    w, h = PAPER[paper]
    fig, ax = new_page(pdf, w, h, "Printing and use", "")
    text = [
        "PRINT AT 100 %, NOT 'FIT TO PAGE'.",
        "",
        "Every sheet has a 100 mm reference bar. Measure it after printing.",
        "If it is not 100 mm, reprint with scaling disabled -- otherwise every",
        "distance you report in the thesis is wrong by that factor.",
        "",
        "Lane width used here: %d mm    Line width: %d mm" % (lane_width, line_width),
        "",
        camera_note,
        "",
        "Bench test procedure",
        "  1. python3 calibrate.py --mode place     (check the ROI sees the lines)",
        "  2. python3 calibrate.py --mode scale     (get pixels per cm)",
        "  3. python3 main.py --no-motors           (slide the sheet left and right;",
        "     the error and the digital twin must follow the sheet)",
        "  4. python3 main.py --record --no-motors  (record each sheet for benchmark.py)",
        "",
        "Print on matt paper if you can. Glossy paper reflects the room lights",
        "into the camera and creates specular highlights that look like lane",
        "markings to any threshold-based detector.",
    ]
    y = h - 30
    for line in text:
        bold = line.isupper() and len(line) > 10
        ax.text(20, y, line, fontsize=9 if bold else 8,
                color="black" if bold else "0.25",
                fontweight="bold" if bold else "normal", va="top")
        y -= 9 if line else 5
    scale_bar(ax, 20, 20)
    pdf.savefig(fig)
    plt.close(fig)


def camera_geometry_note(lane_width):
    """A rough sanity check the student can do with a tape measure."""
    return ("Rule of thumb: the lane must fill roughly half to two thirds of the "
            "image width at the distance the ROI looks at. With a %d mm lane and "
            "the Pi Camera v2, that means about %d-%d cm from the camera to the "
            "band it is looking at." % (lane_width,
                                        int(lane_width * 0.09),
                                        int(lane_width * 0.16)))


def main():
    ap = argparse.ArgumentParser(description="Printable lane test material")
    ap.add_argument("--bench", action="store_true", help="bench test sheets")
    ap.add_argument("--calibration", action="store_true", help="calibration sheet")
    ap.add_argument("--tiles", type=int, default=0, help="number of floor tiles")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--paper", default="a4", choices=list(PAPER))
    ap.add_argument("--lane-width", type=float, default=180.0,
                    help="centre-to-centre lane width in mm")
    ap.add_argument("--line-width", type=float, default=15.0,
                    help="thickness of each lane line in mm")
    ap.add_argument("--out", default="printables")
    args = ap.parse_args()

    if not (args.bench or args.calibration or args.tiles or args.all):
        args.all = True
    os.makedirs(args.out, exist_ok=True)

    if args.all or args.bench:
        path = os.path.join(args.out, "bench_sheets_%s.pdf" % args.paper)
        with PdfPages(path) as pdf:
            instructions_page(pdf, args.paper, args.lane_width, args.line_width,
                              camera_geometry_note(args.lane_width))
            n = bench_sheets(pdf, args.paper, args.lane_width, args.line_width)
        print("  %s  (%d test sheets + instructions)" % (path, n))

    if args.all or args.calibration:
        path = os.path.join(args.out, "calibration_%s.pdf" % args.paper)
        with PdfPages(path) as pdf:
            calibration_sheet(pdf, args.paper, args.lane_width, args.line_width)
        print("  %s" % path)

    count = args.tiles or (6 if args.all else 0)
    if count:
        path = os.path.join(args.out, "floor_tiles_%s.pdf" % args.paper)
        with PdfPages(path) as pdf:
            n = floor_tiles(pdf, args.paper, args.lane_width, args.line_width, count)
        print("  %s  (%d tiles)" % (path, n))

    print("\nPrint at 100 %% scaling and check the reference bar with a ruler.\n")


if __name__ == "__main__":
    main()
