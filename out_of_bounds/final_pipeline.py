
"""
FINAL OUT-OF-BOUNDS PIPELINE
More Than A Hack 2027

Run:
    python -m out_of_bounds.final_pipeline

Uses existing:
    out_of_bounds.pipeline.detect_out_of_bounds

Outputs:
    out_of_bounds/output/events.csv
    out_of_bounds/output/kick_candidates.csv
    out_of_bounds/output/tentative_dead_ball_intervals.csv

IMPORTANT:
    Dead-ball intervals are heuristic candidates,
    NOT verified D labels.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from kloppy import metrica, sportscode

from games import asset_paths
from out_of_bounds.pipeline import detect_out_of_bounds


# ============================================================
# 1. CONFIGURATION
# ============================================================

FPS = 25

PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0

HALF_RANGES = [
    (11185, 90651),
    (110808, 190197),
]

OUTPUT_DIR = Path("out_of_bounds/output")

MAX_RESTART_SEARCH_SECONDS = 30

MIN_SPEED_BEFORE = 1.0
MIN_SPEED_AFTER = 3.0

MAX_PLAYER_DISTANCE = 1.0

LOOKAHEAD_FRAMES = 10
GROUP_GAP_FRAMES = 5


# ============================================================
# 2. BALL SPEED
# ============================================================

def calculate_speed(window):
    """
    Calculate ball speed in metres/second.

    Velocity is only calculated when two consecutive
    video frames contain valid ball coordinates.
    """

    frame = window["video_frame"].to_numpy(dtype=int)

    x = window["ball_x"].to_numpy(dtype=float)
    y = window["ball_y"].to_numpy(dtype=float)

    speed = np.full(len(window), np.nan)

    if len(window) < 2:
        return speed

    delta_frames = np.diff(frame)

    valid = (
        (delta_frames == 1)
        & np.isfinite(x[1:])
        & np.isfinite(x[:-1])
        & np.isfinite(y[1:])
        & np.isfinite(y[:-1])
    )

    vx = np.diff(x) * PITCH_LENGTH * FPS
    vy = np.diff(y) * PITCH_WIDTH * FPS

    speed[1:] = np.where(
        valid,
        np.hypot(vx, vy),
        np.nan
    )

    return speed


# ============================================================
# 3. FIND POSSIBLE KICKS
# ============================================================

def detect_kicks(tracking, start_frame, end_frame):
    """
    Detect possible kicks using:
    - ball speed transition
    - player proximity
    - tracking continuity

    Returns grouped kick candidates.
    """

    window = tracking.loc[
        tracking["video_frame"].between(
            start_frame - 1,
            end_frame + LOOKAHEAD_FRAMES
        )
    ].copy()

    window = (
        window.sort_values("video_frame")
        .reset_index(drop=True)
    )

    if len(window) < LOOKAHEAD_FRAMES + 2:
        return []

    frames = window["video_frame"].to_numpy(dtype=int)
    speed = calculate_speed(window)

    player_x_cols = [
        col for col in window.columns
        if col.endswith("_x")
        and col != "ball_x"
        and col[:-2] + "_y" in window.columns
    ]

    if not player_x_cols:
        return []

    player_y_cols = [
        col[:-2] + "_y"
        for col in player_x_cols
    ]

    ball_x = window["ball_x"].to_numpy(dtype=float)
    ball_y = window["ball_y"].to_numpy(dtype=float)

    player_x = window[player_x_cols].to_numpy(dtype=float)
    player_y = window[player_y_cols].to_numpy(dtype=float)

    distances = np.hypot(
        (player_x - ball_x[:, None]) * PITCH_LENGTH,
        (player_y - ball_y[:, None]) * PITCH_WIDTH
    )

    distances = np.where(
        np.isfinite(distances),
        distances,
        np.inf
    )

    nearest_distances = distances.min(axis=1)
    nearest_indices = distances.argmin(axis=1)

    candidates = []

    for i in range(1, len(window) - LOOKAHEAD_FRAMES):

        frame = int(frames[i])
        future_index = i + LOOKAHEAD_FRAMES

        # Stay inside requested search window
        if not (start_frame <= frame <= end_frame):
            continue

        # Require continuous frame numbering
        if frames[i - 1] != frame - 1:
            continue

        if (
            frames[future_index]
            != frame + LOOKAHEAD_FRAMES
        ):
            continue

        before = speed[i - 1]
        after = speed[future_index]

        if not (
            np.isfinite(before)
            and np.isfinite(after)
        ):
            continue

        if before > MIN_SPEED_BEFORE:
            continue

        if after < MIN_SPEED_AFTER:
            continue

        nearest = nearest_distances[i]

        if nearest > MAX_PLAYER_DISTANCE:
            continue

        player_index = int(nearest_indices[i])

        candidates.append({
            "frame": frame,
            "player_track":
                player_x_cols[player_index][:-2],
            "nearest_player_m": float(nearest),
            "speed_before_mps": float(before),
            "speed_after_mps": float(after),
        })

    # Group nearby candidate frames into one kick event
    groups = []

    for candidate in candidates:

        if (
            not groups
            or candidate["frame"]
            - groups[-1][-1]["frame"] > GROUP_GAP_FRAMES
        ):
            groups.append([candidate])

        else:
            groups[-1].append(candidate)

    kick_events = []

    for group in groups:

        representative = min(
            group,
            key=lambda row: row["nearest_player_m"]
        )

        kick_events.append({
            "start_frame": group[0]["frame"],
            "end_frame": group[-1]["frame"],
            "representative_frame":
                representative["frame"],
            "player_track":
                representative["player_track"],
            "nearest_player_m":
                representative["nearest_player_m"],
            "candidate_frames": len(group),
        })

    return kick_events


# ============================================================
# 4. LOAD SMART TAGGING
# ============================================================

def load_restart_tags(paths):
    """
    Load set-piece annotations.

    Sportscode tags represent broad intervals.
    They are not exact restart timestamps.
    """

    columns = [
        "tag_start",
        "tag_end",
        "restart_type",
        "tag_team",
    ]

    path = paths.get("smart_tagging")

    if path is None or not Path(path).is_file():

        print("WARNING: Smart Tagging unavailable.")

        return pd.DataFrame(columns=columns)

    try:

        events = sportscode.load(path).to_df()

        events = events.loc[
            events["code"]
            .astype(str)
            .str.upper()
            .eq("SET PIECES")
        ].copy()

        if events.empty:
            return pd.DataFrame(columns=columns)

        events["tag_start"] = (
            events["timestamp"]
            .dt.total_seconds() * FPS
        ).round().astype(int)

        events["tag_end"] = (
            events["end_timestamp"]
            .dt.total_seconds() * FPS
        ).round().astype(int)

        if "Type" in events.columns:
            events["restart_type"] = (
                events["Type"]
                .fillna("Unknown")
                .astype(str)
            )
        else:
            events["restart_type"] = "Unknown"

        if "Team" in events.columns:
            events["tag_team"] = (
                events["Team"]
                .fillna("Unknown")
                .astype(str)
            )
        else:
            events["tag_team"] = "Unknown"

        return events[columns].reset_index(drop=True)

    except Exception as error:

        print("WARNING: Smart Tagging error:", error)

        return pd.DataFrame(columns=columns)


# ============================================================
# 5. ASSOCIATE EXITS WITH RESTART CANDIDATES
# ============================================================

def analyze_restarts(tracking, exits, tags):
    """
    Associate observed ball exits with possible kicks.

    A tentative interval requires:
    - observed full-ball exit
    - possible player-adjacent kick
    - overlapping set-piece tag

    This is NOT proof of referee restart timing.
    """

    kick_results = []
    interval_results = []

    exits = (
        exits.sort_values("start_frame")
        .reset_index(drop=True)
    )

    for index, event in exits.iterrows():

        exit_start = int(event["start_frame"])

        full_exit = event["full_ball_exit_frame"]

        if pd.isna(full_exit):
            continue

        full_exit = int(full_exit)

        # Find current graded half
        current_half = None

        for half_start, half_end in HALF_RANGES:

            if half_start <= exit_start <= half_end:
                current_half = (half_start, half_end)
                break

        if current_half is None:
            continue

        half_end = current_half[1]

        # Avoid searching beyond the next outside event
        if index + 1 < len(exits):

            next_exit = int(
                exits.iloc[index + 1]["start_frame"]
            )

        else:
            next_exit = half_end + 1

        search_end = min(
            half_end,
            full_exit
            + MAX_RESTART_SEARCH_SECONDS * FPS,
            next_exit - 1
        )

        # Do not consider kicks while the ball
        # is still observed outside.
        search_start = max(
            full_exit + 1,
            int(event["end_frame"]) + 1
        )

        if search_end < search_start:
            continue

        kicks = detect_kicks(
            tracking,
            search_start,
            search_end
        )

        relevant_tags = tags.loc[
            (tags["tag_end"] >= full_exit)
            & (tags["tag_start"] <= search_end)
        ]

        selected_restart = None

        for kick in kicks:

            kick_frame = kick["representative_frame"]

            overlapping_tags = relevant_tags.loc[
                (relevant_tags["tag_start"] <= kick_frame)
                & (relevant_tags["tag_end"] >= kick_frame)
            ]

            tag_supported = not overlapping_tags.empty

            tag_types = ";".join(
                sorted(
                    set(
                        overlapping_tags[
                            "restart_type"
                        ].astype(str)
                    )
                )
            )

            kick_results.append({
                "exit_start_frame": exit_start,
                **kick,
                "tag_supported": tag_supported,
                "tag_types": tag_types,
            })

            # Select the earliest tag-supported candidate
            if selected_restart is None and tag_supported:
                selected_restart = kick

        # Conservative candidate interval generation
        if (
            selected_restart is not None
            and event["exit_reliability"]
            == "observed_full_exit"
        ):

            restart_frame = int(
                selected_restart["start_frame"]
            )

            if restart_frame > full_exit:

                interval_results.append({
                    "start_frame": full_exit,
                    "end_frame_exclusive": restart_frame,
                    "restart_candidate_frame": restart_frame,
                    "exit_boundary": event["boundary"],
                    "basis":
                        "observed_exit+kick+set_piece",
                    "status":
                        "tentative_not_ground_truth",
                })

    kick_columns = [
        "exit_start_frame",
        "start_frame",
        "end_frame",
        "representative_frame",
        "player_track",
        "nearest_player_m",
        "candidate_frames",
        "tag_supported",
        "tag_types",
    ]

    interval_columns = [
        "start_frame",
        "end_frame_exclusive",
        "restart_candidate_frame",
        "exit_boundary",
        "basis",
        "status",
    ]

    kicks_df = pd.DataFrame(
        kick_results,
        columns=kick_columns
    )

    intervals_df = pd.DataFrame(
        interval_results,
        columns=interval_columns
    )

    return kicks_df, intervals_df


def build_frame_features(tracking, exits, kicks, intervals):
    """Create frame-aligned, weak out-of-bounds features for model training."""

    frames = tracking["video_frame"].to_numpy(dtype=int)
    features = pd.DataFrame({"video_frame": frames})
    features["out_of_bounds_event"] = 0
    features["ball_outside_observed"] = 0
    features["full_ball_exit"] = 0
    features["restart_candidate"] = 0
    features["dead_ball_candidate"] = 0
    features["frames_since_last_exit"] = -1
    features["frames_until_next_exit"] = -1
    features["frames_since_last_restart"] = -1
    features["frames_until_next_restart"] = -1
    features["nearest_exit_distance_frames"] = -1
    features["nearest_restart_distance_frames"] = -1
    features["exit_boundary"] = ""
    features["exit_reliability"] = ""

    frame_index = pd.Series(
        np.arange(len(features)),
        index=features["video_frame"],
    )

    for _, event in exits.iterrows():
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        selected = frame_index.loc[
            frame_index.index.to_series().between(start, end)
        ].to_numpy()

        if len(selected) == 0:
            continue

        features.loc[selected, "out_of_bounds_event"] = 1
        features.loc[selected, "ball_outside_observed"] = 1
        features.loc[selected, "exit_boundary"] = str(event["boundary"])
        features.loc[selected, "exit_reliability"] = str(
            event["exit_reliability"]
        )

        full_exit = event["full_ball_exit_frame"]
        if pd.notna(full_exit):
            full_exit_index = frame_index.get(int(full_exit))
            if full_exit_index is not None:
                features.loc[full_exit_index, "full_ball_exit"] = 1

    for _, kick in kicks.iterrows():
        start = int(kick["start_frame"])
        end = int(kick["end_frame"])
        selected = frame_index.loc[
            frame_index.index.to_series().between(start, end)
        ].to_numpy()
        if len(selected):
            features.loc[selected, "restart_candidate"] = 1

    for _, interval in intervals.iterrows():
        start = int(interval["start_frame"])
        end = int(interval["end_frame_exclusive"]) - 1
        selected = frame_index.loc[
            frame_index.index.to_series().between(start, end)
        ].to_numpy()
        if len(selected):
            features.loc[selected, "dead_ball_candidate"] = 1

    exit_frames = sorted(
        set(exits["start_frame"].dropna().astype(int))
    )
    restart_frames = sorted(
        set(kicks["representative_frame"].dropna().astype(int))
    )

    def add_frame_distances(event_frames, prefix):
        if not event_frames:
            return

        event_array = np.asarray(event_frames, dtype=int)
        positions = np.searchsorted(event_array, frames)
        previous_exists = positions > 0
        next_exists = positions < len(event_array)

        previous = np.full(len(frames), -1, dtype=int)
        previous[previous_exists] = (
            frames[previous_exists]
            - event_array[positions[previous_exists] - 1]
        )

        following = np.full(len(frames), -1, dtype=int)
        following[next_exists] = (
            event_array[positions[next_exists]]
            - frames[next_exists]
        )

        nearest = np.full(len(frames), -1, dtype=int)
        both = previous_exists & next_exists
        nearest[previous_exists & ~next_exists] = previous[
            previous_exists & ~next_exists
        ]
        nearest[~previous_exists & next_exists] = following[
            ~previous_exists & next_exists
        ]
        nearest[both] = np.minimum(previous[both], following[both])

        features[f"frames_since_last_{prefix}"] = previous
        features[f"frames_until_next_{prefix}"] = following
        features[f"nearest_{prefix}_distance_frames"] = nearest

    add_frame_distances(exit_frames, "exit")
    add_frame_distances(restart_frames, "restart")

    return features


# ============================================================
# 6. FINAL MAIN PIPELINE
# ============================================================

def main():

    print("===== FINAL OUT-OF-BOUNDS PIPELINE =====")

    paths = asset_paths()

    # Load full-match tracking
    print("\n[1/5] Loading match tracking...")

    dataset = metrica.load_tracking_epts(
        meta_data=paths["atd_meta"],
        raw_data=paths["atd_raw"]
    )

    tracking = dataset.to_df()

    tracking["video_frame"] = (
        tracking["frame_id"] - 1
    )

    if tracking["video_frame"].duplicated().any():
        raise ValueError(
            "Duplicate video_frame values detected"
        )

    print("Total frames:", len(tracking))

    # Detect out-of-bounds events
    print("\n[2/5] Detecting ball exits...")

    exits = detect_out_of_bounds(
        tracking,
        half_ranges=HALF_RANGES
    )

    print("Exit candidates:", len(exits))

    print("\nReliability distribution:")
    print(exits["exit_reliability"].value_counts())

    # Load tactical annotations
    print("\n[3/5] Loading Smart Tagging...")

    tags = load_restart_tags(paths)

    print("Set-piece tags:", len(tags))

    # Detect restart candidates
    print("\n[4/5] Detecting possible restarts...")

    kicks, intervals = analyze_restarts(
        tracking,
        exits,
        tags
    )

    print("Kick candidates:", len(kicks))

    print(
        "Tentative dead-ball intervals:",
        len(intervals)
    )

    # Export results
    print("\n[5/5] Exporting results...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    exits.to_csv(
        OUTPUT_DIR / "events.csv",
        index=False
    )

    kicks.to_csv(
        OUTPUT_DIR / "kick_candidates.csv",
        index=False
    )

    intervals.to_csv(
        OUTPUT_DIR / "tentative_dead_ball_intervals.csv",
        index=False
    )

    frame_features = build_frame_features(
        tracking,
        exits,
        kicks,
        intervals,
    )
    frame_features.to_csv(
        OUTPUT_DIR / "frame_features.csv",
        index=False
    )

    print("\n===== PIPELINE COMPLETED =====")

    print("Output folder:", OUTPUT_DIR)

    print("\nGenerated files:")
    print("- events.csv")
    print("- kick_candidates.csv")
    print("- tentative_dead_ball_intervals.csv")
    print("- frame_features.csv")

    print(
        "\nWARNING: Tentative intervals are not "
        "verified D labels. Do not automatically "
        "override possession predictions."
    )


if __name__ == "__main__":
    main()
