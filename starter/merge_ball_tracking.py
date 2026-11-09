"""Create a complete repaired ball trajectory for the tactical video.

Precedence is deliberately conservative:
1. Keep every observed ATD position.
2. Use a YOLO pitch position only where ATD is missing and confidence is sufficient.
3. Interpolate remaining internal gaps up to the configured maximum length.
4. Leave longer or edge gaps unresolved.

The output is derived data and should not be committed or redistributed.
"""

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from ball_interpolation import interpolate_gaps, load_ball_positions  # noqa: E402
from games import GAME, GAMES, require, source_frames  # noqa: E402


def load_yolo_positions(path, n_frames, confidence_threshold):
    """Return YOLO x/y arrays in metres and a mask of accepted detections."""
    x = np.full(n_frames, np.nan, dtype=np.float64)
    y = np.full(n_frames, np.nan, dtype=np.float64)
    accepted = np.zeros(n_frames, dtype=bool)
    if not path:
        return x, y, accepted

    table = pd.read_csv(path)
    required = {"video_frame", "x_m", "y_m", "confidence"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError("{}: missing columns {}".format(path, missing))

    for row in table.itertuples(index=False):
        frame = int(row.video_frame)
        if not 0 <= frame < n_frames:
            continue
        if not np.isfinite(row.x_m) or not np.isfinite(row.y_m):
            continue
        if not np.isfinite(row.confidence) or row.confidence < confidence_threshold:
            continue
        if not (0.0 <= row.x_m <= GAMES[GAME]["pitch"][0]
                and 0.0 <= row.y_m <= GAMES[GAME]["pitch"][1]):
            continue
        x[frame] = float(row.x_m)
        y[frame] = float(row.y_m)
        accepted[frame] = True
    return x, y, accepted


def write_output(path, x, y, source):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "ball_x_m", "ball_y_m", "source"])
        for frame in range(len(x)):
            writer.writerow([
                frame,
                "" if not np.isfinite(x[frame]) else "{:.6f}".format(x[frame]),
                "" if not np.isfinite(y[frame]) else "{:.6f}".format(y[frame]),
                source[frame],
            ])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yolo", default=None,
                        help="optional YOLO CSV with video_frame, x_m, y_m, confidence")
    parser.add_argument("--confidence", type=float, default=0.40,
                        help="minimum YOLO confidence")
    parser.add_argument("--max-gap", type=int, default=35,
                        help="maximum remaining gap to interpolate")
    parser.add_argument("--output", default=os.path.join(
        ROOT, "processed", "tactical_ball_repaired.csv"))
    args = parser.parse_args()

    if args.max_gap < 0:
        parser.error("--max-gap must be non-negative")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")

    n_frames = source_frames("tactical", GAME)
    raw_path = require(["atd_raw"], GAME, source="tactical")[0]
    ball_x, ball_y = load_ball_positions(raw_path, n_frames, GAMES[GAME]["pitch"])
    observed = np.isfinite(ball_x) & np.isfinite(ball_y)

    source = np.full(n_frames, "unresolved", dtype="<U12")
    source[observed] = "ATD"

    yolo_x, yolo_y, yolo_ok = load_yolo_positions(
        args.yolo, n_frames, args.confidence
    )
    yolo_fill = ~observed & yolo_ok
    ball_x[yolo_fill] = yolo_x[yolo_fill]
    ball_y[yolo_fill] = yolo_y[yolo_fill]
    source[yolo_fill] = "YOLO"

    before_interpolation = np.isfinite(ball_x) & np.isfinite(ball_y)
    ball_x, ball_y, interpolation_mask = interpolate_gaps(
        ball_x, ball_y, max_gap=args.max_gap
    )
    interpolation_fill = interpolation_mask & ~before_interpolation
    source[interpolation_fill] = "interpolated"

    write_output(args.output, ball_x, ball_y, source)
    counts = {name: int(np.sum(source == name)) for name in (
        "ATD", "YOLO", "interpolated", "unresolved"
    )}
    print("wrote: {}".format(args.output))
    print("frames: {} (0..{})".format(n_frames, n_frames - 1))
    print("sources: {}".format(counts))
    print("YOLO accepted: {} (threshold {:.2f})".format(
        int(yolo_ok.sum()), args.confidence
    ))


if __name__ == "__main__":
    main()
