
import pandas as pd


def detect_outside_frames(ball):
    """
    Detect frames where the ball is outside
    the normalized pitch boundaries.
    """

    valid = ball[["ball_x", "ball_y"]].notna().all(axis=1)

    outside = valid & (
        (ball["ball_x"] < 0) |
        (ball["ball_x"] > 1) |
        (ball["ball_y"] < 0) |
        (ball["ball_y"] > 1)
    )

    return outside


def group_outside_events(ball):
    """
    Group consecutive outside frames into events.
    """

    ball = ball.sort_values("video_frame").copy()

    outside = detect_outside_frames(ball)

    outside_frames = ball.loc[outside, "video_frame"]

    events = []

    if outside_frames.empty:
        return pd.DataFrame(
            columns=["start_frame", "end_frame", "duration_frames"]
        )

    start = int(outside_frames.iloc[0])
    previous = start

    for frame in outside_frames.iloc[1:]:
        frame = int(frame)

        if frame != previous + 1:
            events.append({
                "start_frame": start,
                "end_frame": previous,
                "duration_frames": previous - start + 1
            })
            start = frame

        previous = frame

    events.append({
        "start_frame": start,
        "end_frame": previous,
        "duration_frames": previous - start + 1
    })

    return pd.DataFrame(events)


import numpy as np

def calculate_ball_velocity(ball):
    ball = ball.sort_values("video_frame").copy()

    # Time difference between frames
    dt = ball["video_frame"].diff() / 25.0

    # Convert normalized position differences to metres
    dx = ball["ball_x"].diff() * 105.3
    dy = ball["ball_y"].diff() * 68.0

    ball["vx"] = dx / dt
    ball["vy"] = dy / dt

    ball["speed"] = np.sqrt(
        ball["vx"] ** 2 + ball["vy"] ** 2
    )

    # Negative vx means movement toward x=0
    # Positive vx means movement toward x=1

    return ball


import numpy as np


def detect_kick_candidates(
    tracking,
    min_speed_before=1.0,
    min_speed_after=3.0,
    max_player_distance=1.0,
    lookahead=10
):
    """
    Detect possible ball kicks from changes in speed
    and proximity to tracked players.

    These are heuristic candidate thresholds, not
    calibrated or validated restart decisions.
    """
    df = tracking.sort_values("video_frame").copy()

    # Calculate velocity if not already present
    if "speed" not in df.columns:
        df = calculate_ball_velocity(df)

    player_x_cols = [
        c for c in df.columns
        if c.endswith("_x") and c != "ball_x"
    ]

    candidates = []

    for i in range(1, len(df) - lookahead):
        current = df.iloc[i]
        previous = df.iloc[i - 1]
        future = df.iloc[i + lookahead]

        # Require consecutive observations
        if (
            current["video_frame"] != previous["video_frame"] + 1
            or future["video_frame"] !=
            current["video_frame"] + lookahead
        ):
            continue

        # Low speed before and stronger movement afterward
        if not (
            np.isfinite(previous["speed"])
            and np.isfinite(future["speed"])
            and previous["speed"] <= min_speed_before
            and future["speed"] >= min_speed_after
        ):
            continue

        bx, by = current["ball_x"], current["ball_y"]

        if not np.isfinite([bx, by]).all():
            continue

        nearest = float("inf")

        for x_col in player_x_cols:
            y_col = x_col[:-2] + "_y"

            if y_col not in df.columns:
                continue

            px, py = current[x_col], current[y_col]

            if not np.isfinite([px, py]).all():
                continue

            distance = np.hypot(
                (px - bx) * 105.3,
                (py - by) * 68.0
            )

            nearest = min(nearest, distance)

        if nearest <= max_player_distance:
            candidates.append({
                "frame": int(current["video_frame"]),
                "speed_before": float(previous["speed"]),
                "speed_after": float(future["speed"]),
                "nearest_player_m": float(nearest)
            })

    return candidates

def group_kick_candidates(candidates, max_gap=5):
    """
    Group nearby candidate frames into kick events.
    max_gap is measured in frames.
    """
    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda c: c["frame"]
    )

    groups = []
    current = [candidates[0]]

    for candidate in candidates[1:]:
        previous = current[-1]

        if candidate["frame"] - previous["frame"] <= max_gap:
            current.append(candidate)
        else:
            groups.append(current)
            current = [candidate]

    groups.append(current)

    events = []

    for group in groups:
        closest = min(
            group,
            key=lambda c: c["nearest_player_m"]
        )

        events.append({
            "start_frame": group[0]["frame"],
            "end_frame": group[-1]["frame"],
            "representative_frame": closest["frame"],
            "min_player_distance_m":
                closest["nearest_player_m"],
            "candidate_count": len(group)
        })

    return events
