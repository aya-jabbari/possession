"""Ball trajectory interpolation for missing ball positions (NaN gaps).

The ATD tracking loses the ball on 41% of graded frames -- occlusions, high passes,
crowded penalty boxes. The baseline treats every missing frame as "nobody has the ball"
(label D), which is wrong: in reality the ball is still in play, just invisible to the
tracker.

This module recovers short gaps by interpolating the (x, y) trajectory between the last
known position before the gap and the first known position after it.

Algorithm
---------
1. Extract ball (x, y) from the raw ATD as a float array, NaN where missing.
2. Scan for contiguous NaN runs (gaps).
3. For each gap:
   a. If gap length > MAX_GAP_FRAMES: skip (too uncertain, leave NaN).
   b. Compute the entry velocity (ball speed just before the gap).
   c. Compute the exit velocity (ball speed just after the gap).
   d. If both velocities are >= MIN_SPEED_MS (ball was actually moving, not resting):
      -> use cubic spline interpolation (smooth curved trajectory, good for passes).
   e. Else:
      -> use linear interpolation (ball was slow/static, simpler is better).
4. Return the full (x, y) arrays with gaps filled where possible.

Why these defaults
------------------
- MAX_GAP_FRAMES = 35  ~ 1.4 s at 25 fps. Beyond that, a free kick, corner or throw-in
  could have happened and interpolating across a restart would be wrong.
- MIN_SPEED_MS = 1.5   m/s. Below this the ball is probably stationary (dead ball,
  keeper holding). Cubic spline on two nearly-identical points produces a flat line
  anyway, but linear is cleaner.
- VELOCITY_WINDOW = 4  frames to estimate velocity (~0.16 s). Short enough to be local,
  long enough to suppress single-frame jitter.

Usage
-----
    from starter.ball_interpolation import load_ball_positions, interpolate_gaps

    ball_x, ball_y = load_ball_positions(raw_path, n_frames, pitch)
    ball_x_filled, ball_y_filled, fill_mask = interpolate_gaps(ball_x, ball_y)

    # fill_mask[i] is True where frame i was NaN and has been filled in.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require  # noqa: E402

# ── Tunable parameters ────────────────────────────────────────────────────────

MAX_GAP_FRAMES = 35      # gaps longer than this are left untouched (~1.4 s)
MIN_SPEED_MS   = 1.5     # m/s — minimum speed at entry OR exit to use cubic spline
VELOCITY_WINDOW = 4      # frames used to estimate entry/exit velocity

# Raw ATD line format: "<frame_id>:<players>:<ball>"
FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING   = "N"          # first char of "NaN"


# ── Step 1: load raw ball positions ──────────────────────────────────────────

def load_ball_positions(raw_path, n_frames, pitch):
    """Read ball (x, y) in metres from the raw ATD text file.

    Parameters
    ----------
    raw_path  : str   Path to the _atd.txt file.
    n_frames  : int   Total number of video frames (199319 for BAR-ATH).
    pitch     : tuple (length_m, width_m) — used to convert normalised 0-1 coords.

    Returns
    -------
    ball_x, ball_y : np.ndarray of shape (n_frames,), dtype float64.
        NaN where the ball was not detected.
    """
    length_m, width_m = pitch
    ball_x = np.full(n_frames, np.nan)
    ball_y = np.full(n_frames, np.nan)

    with open(raw_path) as fh:
        for line in fh:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            ball_str = parts[2].strip()
            if ball_str[0] == MISSING:
                continue  # NaN — leave as NaN

            coords = ball_str.split(",")
            try:
                x_norm = float(coords[0])
                y_norm = float(coords[1])
            except (ValueError, IndexError):
                continue

            frame_idx = int(parts[0]) - 1   # frame_id counts from 1
            if 0 <= frame_idx < n_frames:
                ball_x[frame_idx] = x_norm * length_m
                ball_y[frame_idx] = y_norm * width_m

    return ball_x, ball_y


# ── Step 2 & 3: find gaps and interpolate ────────────────────────────────────

def _find_nan_gaps(arr):
    """Return list of (start, end) indices of contiguous NaN runs (inclusive)."""
    nan_mask = np.isnan(arr)
    if not nan_mask.any():
        return []

    gaps = []
    in_gap = False
    start = 0
    for i, is_nan in enumerate(nan_mask):
        if is_nan and not in_gap:
            start = i
            in_gap = True
        elif not is_nan and in_gap:
            gaps.append((start, i - 1))
            in_gap = False
    if in_gap:
        gaps.append((start, len(arr) - 1))

    return gaps


def _velocity_ms(arr, idx, window, forward=True):
    """Estimate speed (m/s) at a boundary frame using a short finite difference.

    Parameters
    ----------
    arr     : 1-D float array (metres)
    idx     : boundary frame (last known before gap, or first known after)
    window  : number of frames to look back (forward=False) or ahead (forward=True)
    forward : True → look forward from idx, False → look backward

    Returns
    -------
    speed : float, NaN if not enough data.
    """
    fps = 25
    if forward:
        segment = arr[idx: idx + window + 1]
    else:
        segment = arr[max(0, idx - window): idx + 1]

    valid = segment[~np.isnan(segment)]
    if len(valid) < 2:
        return np.nan

    # Displacement over the segment (approximate, ignores exact frame positions)
    displacement = abs(valid[-1] - valid[0])
    dt = (len(valid) - 1) / fps
    return displacement / dt if dt > 0 else np.nan


def interpolate_gaps(ball_x, ball_y,
                     max_gap=MAX_GAP_FRAMES,
                     min_speed=MIN_SPEED_MS,
                     vel_window=VELOCITY_WINDOW):
    """Fill short ball trajectory gaps by interpolation.

    Parameters
    ----------
    ball_x, ball_y : np.ndarray  Raw ball positions in metres (NaN where missing).
    max_gap        : int         Maximum gap length to attempt filling (frames).
    min_speed      : float       m/s threshold that triggers cubic spline vs linear.
    vel_window     : int         Frames used to estimate entry/exit speed.

    Returns
    -------
    filled_x, filled_y : np.ndarray  Positions with gaps filled in.
    fill_mask          : np.ndarray  bool, True where a frame was NaN and was filled.
    """
    filled_x = ball_x.copy()
    filled_y = ball_y.copy()
    fill_mask = np.zeros(len(ball_x), dtype=bool)

    gaps = _find_nan_gaps(ball_x)   # x and y share the same NaN pattern

    for gap_start, gap_end in gaps:
        gap_len = gap_end - gap_start + 1

        # ── a. too long → skip ────────────────────────────────────────────────
        if gap_len > max_gap:
            continue

        # ── boundary frames ───────────────────────────────────────────────────
        pre  = gap_start - 1   # last known frame before the gap
        post = gap_end   + 1   # first known frame after the gap

        if pre < 0 or post >= len(ball_x):
            # Gap touches the edge of the recording — cannot interpolate
            continue

        if np.isnan(ball_x[pre]) or np.isnan(ball_x[post]):
            # Boundary itself is NaN (adjacent gap) — skip
            continue

        # ── b & c. entry and exit speeds ──────────────────────────────────────
        entry_speed_x = _velocity_ms(filled_x, pre,  vel_window, forward=False)
        exit_speed_x  = _velocity_ms(filled_x, post, vel_window, forward=True)
        entry_speed_y = _velocity_ms(filled_y, pre,  vel_window, forward=False)
        exit_speed_y  = _velocity_ms(filled_y, post, vel_window, forward=True)

        entry_speed = max(
            entry_speed_x if not np.isnan(entry_speed_x) else 0.0,
            entry_speed_y if not np.isnan(entry_speed_y) else 0.0,
        )
        exit_speed = max(
            exit_speed_x if not np.isnan(exit_speed_x) else 0.0,
            exit_speed_y if not np.isnan(exit_speed_y) else 0.0,
        )

        # Frames to fill
        frames_to_fill = np.arange(gap_start, gap_end + 1)

        # Anchor frames and known values (pre and post)
        anchor_frames = np.array([pre, post])
        anchor_x      = np.array([filled_x[pre], filled_x[post]])
        anchor_y      = np.array([filled_y[pre], filled_y[post]])

        # ── d & e. choose interpolation method ────────────────────────────────
        use_spline = (entry_speed >= min_speed) and (exit_speed >= min_speed)

        if use_spline and gap_len >= 2:
            # Cubic spline — smooth curved arc, appropriate for passes/shots
            try:
                cs_x = CubicSpline(anchor_frames, anchor_x, bc_type="not-a-knot")
                cs_y = CubicSpline(anchor_frames, anchor_y, bc_type="not-a-knot")
                filled_x[frames_to_fill] = cs_x(frames_to_fill)
                filled_y[frames_to_fill] = cs_y(frames_to_fill)
            except Exception:
                # Fallback to linear if spline fails (e.g. single-point degenerate)
                filled_x[frames_to_fill] = np.interp(
                    frames_to_fill, anchor_frames, anchor_x)
                filled_y[frames_to_fill] = np.interp(
                    frames_to_fill, anchor_frames, anchor_y)
        else:
            # Linear interpolation — ball slow or static
            filled_x[frames_to_fill] = np.interp(
                frames_to_fill, anchor_frames, anchor_x)
            filled_y[frames_to_fill] = np.interp(
                frames_to_fill, anchor_frames, anchor_y)

        fill_mask[frames_to_fill] = True

    return filled_x, filled_y, fill_mask


# ── Convenience wrapper ───────────────────────────────────────────────────────

def load_and_interpolate(game=GAME, data_dir=None,
                         max_gap=MAX_GAP_FRAMES,
                         min_speed=MIN_SPEED_MS):
    """One-call helper: load ATD + interpolate ball gaps.

    Returns
    -------
    ball_x, ball_y : filled arrays in metres
    fill_mask      : bool array, True where a frame was interpolated
    pitch          : (length_m, width_m) tuple
    """
    raw_path = require(["atd_raw"], game, data_dir)[0]
    n_frames = GAMES[game]["n_video_frames"]
    pitch    = GAMES[game]["pitch"]

    print("Loading ball positions from ATD...")
    ball_x, ball_y = load_ball_positions(raw_path, n_frames, pitch)

    n_missing_before = int(np.isnan(ball_x).sum())
    print(f"  Ball missing on {n_missing_before:,} / {n_frames:,} frames "
          f"({100*n_missing_before/n_frames:.1f}%)")

    print(f"Interpolating gaps (max {max_gap} frames, min speed {min_speed} m/s)...")
    ball_x, ball_y, fill_mask = interpolate_gaps(
        ball_x, ball_y, max_gap=max_gap, min_speed=min_speed)

    n_filled = int(fill_mask.sum())
    n_missing_after = int(np.isnan(ball_x).sum())
    print(f"  Filled {n_filled:,} frames  |  still missing: {n_missing_after:,} "
          f"({100*n_missing_after/n_frames:.1f}%)")

    return ball_x, ball_y, fill_mask, pitch


# ── CLI smoke-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ball_x, ball_y, fill_mask, pitch = load_and_interpolate()

    # Show a sample of filled gaps
    gap_indices = np.where(fill_mask)[0]
    if gap_indices.size:
        print(f"\nSample filled frames (video frame, x_m, y_m):")
        for idx in gap_indices[:10]:
            print(f"  frame {idx:6d}  x={ball_x[idx]:6.2f} m  y={ball_y[idx]:6.2f} m")
    else:
        print("\nNo gaps were filled (data file may not be present).")
