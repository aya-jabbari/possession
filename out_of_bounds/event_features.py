
import numpy as np
import pandas as pd

from out_of_bounds.detector import group_outside_events

PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0
BALL_RADIUS = 0.11


def analyze_outside_events(ball):
    """
    Extract geometric and kinematic features for each
    outside event.

    Requires video_frame, ball_x, ball_y, vx, vy.
    """

    ball = ball.sort_values("video_frame").copy()

    events = group_outside_events(ball)

    results = []

    for _, event in events.iterrows():
        start = int(event["start_frame"])
        end = int(event["end_frame"])

        segment = ball.loc[
            ball["video_frame"].between(start, end)
        ]

        first = segment.iloc[0]

        x = first["ball_x"]
        y = first["ball_y"]

        # Identify the crossed boundary
        if x < 0:
            boundary = "left_goal_line"
            outward_speed = -first["vx"]
        elif x > 1:
            boundary = "right_goal_line"
            outward_speed = first["vx"]
        elif y < 0:
            boundary = "bottom_touchline"
            outward_speed = -first["vy"]
        elif y > 1:
            boundary = "top_touchline"
            outward_speed = first["vy"]
        else:
            boundary = "unknown"
            outward_speed = np.nan

        # Calculate maximum outside distance in metres
        outside_distances = np.maximum.reduce([
            -segment["ball_x"].to_numpy() * PITCH_LENGTH,
            (segment["ball_x"].to_numpy() - 1) * PITCH_LENGTH,
            -segment["ball_y"].to_numpy() * PITCH_WIDTH,
            (segment["ball_y"].to_numpy() - 1) * PITCH_WIDTH,
            np.zeros(len(segment))
        ])

        max_distance = float(np.max(outside_distances))

        # Count consecutive missing frames immediately afterward
        following = ball.loc[
            ball["video_frame"] > end
        ]

        missing_gap = 0

        for _, row in following.iterrows():
            if (
                pd.notna(row["ball_x"]) and
                pd.notna(row["ball_y"])
            ):
                break

            missing_gap += 1

        results.append({
            "start_frame": start,
            "end_frame": end,
            "boundary": boundary,
            "duration_frames": end - start + 1,
            "outward_speed_mps": outward_speed,
            "max_outside_distance_m": max_distance,
            "fully_outside_observed":
                max_distance > BALL_RADIUS,
            "missing_gap_after": missing_gap
        })

    return pd.DataFrame(results)

def classify_exit_evidence(events):
    """
    Categorize boundary-crossing evidence.

    These classifications describe evidence quality,
    not confirmed referee decisions.
    """

    events = events.copy()

    def classify(row):
        if not row["fully_outside_observed"]:
            return "uncertain"

        speed = row["outward_speed_mps"]

        if pd.isna(speed):
            return "geometry_only"

        if speed > 0:
            return "strong_candidate"

        return "inconsistent_motion"

    events["evidence_level"] = events.apply(
        classify,
        axis=1
    )

    return events

def validate_boundary_crossings(ball, features):
    """
    Check whether an outside event has an observed
    inside-to-outside transition.

    This is geometric evidence only, not a confirmed
    referee stoppage.
    """

    ball = ball.sort_values("video_frame").copy()
    features = features.copy()

    # Fast frame-index lookup
    positions = ball.set_index("video_frame")

    crossing_status = []

    for _, event in features.iterrows():
        start = int(event["start_frame"])

        previous_frame = start - 1

        if previous_frame not in positions.index:
            crossing_status.append("no_previous_frame")
            continue

        previous = positions.loc[previous_frame]
        current = positions.loc[start]

        previous_valid = (
            pd.notna(previous["ball_x"]) and
            pd.notna(previous["ball_y"])
        )

        if not previous_valid:
            crossing_status.append("appeared_outside")
            continue

        x = previous["ball_x"]
        y = previous["ball_y"]

        previous_inside = (
            0 <= x <= 1 and
            0 <= y <= 1
        )

        if previous_inside:
            crossing_status.append("observed_crossing")
        else:
            crossing_status.append("already_outside")

    features["crossing_status"] = crossing_status

    return features

def find_full_ball_exit_frames(ball, features):
    """
    Find the first observed frame where the entire
    ball is outside the event's primary boundary.

    This does not establish the exact physical
    crossing time between video frames.
    """

    ball = ball.sort_values("video_frame").copy()
    features = features.copy()

    radius_x = BALL_RADIUS / PITCH_LENGTH
    radius_y = BALL_RADIUS / PITCH_WIDTH

    exit_frames = []

    for _, event in features.iterrows():
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        boundary = event["boundary"]

        segment = ball.loc[
            ball["video_frame"].between(start, end)
        ]

        if boundary == "left_goal_line":
            fully_outside = segment["ball_x"] < -radius_x

        elif boundary == "right_goal_line":
            fully_outside = segment["ball_x"] > 1 + radius_x

        elif boundary == "bottom_touchline":
            fully_outside = segment["ball_y"] < -radius_y

        elif boundary == "top_touchline":
            fully_outside = segment["ball_y"] > 1 + radius_y

        else:
            exit_frames.append(pd.NA)
            continue

        crossing = segment.loc[fully_outside]

        if crossing.empty:
            exit_frames.append(pd.NA)
        else:
            exit_frames.append(
                int(crossing["video_frame"].iloc[0])
            )

    features["full_ball_exit_frame"] = pd.array(
        exit_frames, dtype="Int64"
    )

    return features

def add_exit_reliability(features):
    """
    Separate observed crossings from uncertain
    first-outside observations.
    """

    features = features.copy()

    def classify(row):
        if pd.isna(row["full_ball_exit_frame"]):
            return "no_full_crossing"

        if row["crossing_status"] == "appeared_outside":
            return "exit_time_unknown"

        if (
            row["crossing_status"] == "observed_crossing"
            and pd.notna(row["outward_speed_mps"])
            and row["outward_speed_mps"] > 0
        ):
            return "observed_full_exit"

        return "needs_review"

    features["exit_reliability"] = features.apply(
        classify, axis=1
    )

    return features

def validate_full_exit_continuity(ball, features):
    """
    Validate continuous observations from the first
    centre-outside frame through the first full-ball-
    outside frame.

    This does not confirm an actual referee decision.
    """

    ball_indexed = (
        ball.sort_values("video_frame")
        .set_index("video_frame")
    )

    features = features.copy()
    continuity = []

    for _, event in features.iterrows():
        exit_frame = event["full_ball_exit_frame"]

        if pd.isna(exit_frame):
            continuity.append(False)
            continue

        if event["crossing_status"] != "observed_crossing":
            continuity.append(False)
            continue

        start = int(event["start_frame"])
        end = int(exit_frame)

        expected_frames = end - start + 1

        segment = ball_indexed.loc[start:end]

        valid_positions = (
            segment["ball_x"].notna()
            & segment["ball_y"].notna()
        )

        continuous = (
            len(segment) == expected_frames
            and valid_positions.all()
            and segment.index.is_unique
        )

        continuity.append(bool(continuous))

    features["full_exit_continuous"] = continuity

    # Preserve historical evidence while distinguishing
    # cases with incomplete temporal observations.
    features.loc[
        (features["exit_reliability"] == "observed_full_exit")
        & (~features["full_exit_continuous"]),
        "exit_reliability"
    ] = "continuity_unverified"

    return features
