"""
Step 1: Feature Engineering
===========================
Extract per-frame features from ATD tracking + Smart Tagging for the GBDT model.

Features extracted:
  - Ball-player geometry (nearest player distances, counts within radius)
  - Team formation (centroid, spread, convex hull)
  - Ball kinematics (speed, acceleration, direction)
  - Smart Tagging priors (set piece active, team, type)
  - Missing-data indicators (ball visible, player count, gap length)
  - Temporal context (rolling statistics)

Usage:
    python -m pipeline.features
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES, asset_paths, require, half_video_blocks  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PITCH_LENGTH = GAMES[GAME]["pitch"][0]  # 105.3
PITCH_WIDTH = GAMES[GAME]["pitch"][1]   # 68.0
FPS = GAMES[GAME]["fps"]                # 25
N_FRAMES = GAMES[GAME]["n_video_frames"]  # 199319

HALF_BLOCKS = half_video_blocks(GAME)  # [(1, 11185, 90651), (2, 110808, 190197)]

# ATD file format
FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING_CHAR = "N"  # first char of "NaN"


# ---------------------------------------------------------------------------
# 1. Parse team assignments from ATD metadata
# ---------------------------------------------------------------------------

def load_team_assignments(meta_path):
    """Return array: team_of_track[slot] = 0 (home) or 1 (away)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([0 if tid == home_id else 1 for tid in slots], dtype=np.int8)


# ---------------------------------------------------------------------------
# 2. Parse raw ATD into structured arrays (fast, numpy-only)
# ---------------------------------------------------------------------------

def parse_atd_raw(raw_path, team_of_track, n_frames):
    """
    Parse the raw ATD text file into per-frame arrays.

    Returns dict with arrays of shape (n_frames,) or (n_frames, n_tracks):
      - ball_x, ball_y: ball position (NaN when missing)
      - ball_visible: boolean
      - player_x, player_y: (n_frames, n_tracks) player positions
      - player_speed: (n_frames, n_tracks) player speeds
      - player_team: (n_tracks,) team assignment
    """
    n_tracks = len(team_of_track)

    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    ball_speed_raw = np.full(n_frames, np.nan, dtype=np.float32)

    player_x = np.full((n_frames, n_tracks), np.nan, dtype=np.float32)
    player_y = np.full((n_frames, n_tracks), np.nan, dtype=np.float32)
    player_speed = np.full((n_frames, n_tracks), np.nan, dtype=np.float32)

    print(f"  Parsing {os.path.basename(raw_path)} ({n_frames} frames)...")

    with open(raw_path) as f:
        for line in f:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue

            frame_id = int(parts[0])
            vf = frame_id - 1  # video frame = frame_id - 1

            if vf < 0 or vf >= n_frames:
                continue

            # Ball
            ball_str = parts[2].strip()
            if ball_str[0] != MISSING_CHAR:
                fields = ball_str.split(",")
                ball_x[vf] = float(fields[0])
                ball_y[vf] = float(fields[1])
                if len(fields) > 2:
                    ball_speed_raw[vf] = float(fields[2])

            # Players
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if slot >= n_tracks:
                    break
                if group[0] == MISSING_CHAR:
                    continue
                fields = group.split(",")
                player_x[vf, slot] = float(fields[0])
                player_y[vf, slot] = float(fields[1])
                if len(fields) > 2:
                    player_speed[vf, slot] = float(fields[2])

    ball_visible = np.isfinite(ball_x)
    print(f"  Ball visible on {ball_visible.sum()} / {n_frames} frames "
          f"({100 * ball_visible.mean():.1f}%)")

    return {
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_speed_raw": ball_speed_raw,
        "ball_visible": ball_visible,
        "player_x": player_x,
        "player_y": player_y,
        "player_speed": player_speed,
        "player_team": team_of_track,
    }


# ---------------------------------------------------------------------------
# 3. Parse Smart Tagging
# ---------------------------------------------------------------------------

def parse_smart_tagging(st_path):
    """
    Parse Smart Tagging XML into per-frame arrays.

    Returns arrays of shape (n_frames,):
      - st_team: 0 (home/Barcelona), 1 (away/Athletic), -1 (unknown/N/A)
      - st_is_set_piece: boolean (frame is inside a SET PIECES interval)
      - st_is_offensive_home: boolean (Barcelona has an offensive tag active)
      - st_is_offensive_away: boolean (Athletic has an offensive tag active)
      - st_phase_code: integer encoding of the active phase (-1 = none)
    """
    try:
        from kloppy import sportscode
        events = sportscode.load(st_path).to_df()
    except Exception as e:
        print(f"  WARNING: Could not load Smart Tagging: {e}")
        return _empty_smart_tagging()

    st_team = np.full(N_FRAMES, -1, dtype=np.int8)
    st_is_set_piece = np.zeros(N_FRAMES, dtype=bool)
    st_is_offensive_home = np.zeros(N_FRAMES, dtype=bool)
    st_is_offensive_away = np.zeros(N_FRAMES, dtype=bool)

    # Offensive phase codes
    OFFENSIVE_CODES = {
        "BUILD UP", "PROGRESSION", "CREATING CHANCES",
        "ATTACKING TRANSITION", "BALL IN FINAL THIRD",
        "BALL IN THE BOX", "LONG BALL",
    }

    for _, row in events.iterrows():
        start_s = row["timestamp"].total_seconds()
        end_s = row["end_timestamp"].total_seconds()
        start_f = int(round(start_s * FPS))
        end_f = int(round(end_s * FPS))

        start_f = max(0, start_f)
        end_f = min(N_FRAMES - 1, end_f)

        code = str(row.get("code", "")).upper()
        team_str = str(row.get("Team", "N/A")).strip()

        # Determine team
        team_code = -1
        if "Barcelona" in team_str:
            team_code = 0
        elif "Atletic" in team_str or "Athletic" in team_str:
            team_code = 1

        # Set team label for this interval (latest tag wins if overlapping)
        if team_code >= 0:
            st_team[start_f:end_f + 1] = team_code

        # Set piece flag
        if code == "SET PIECES":
            st_is_set_piece[start_f:end_f + 1] = True

        # Offensive phase flags
        if code in OFFENSIVE_CODES:
            if team_code == 0:
                st_is_offensive_home[start_f:end_f + 1] = True
            elif team_code == 1:
                st_is_offensive_away[start_f:end_f + 1] = True

    return {
        "st_team": st_team,
        "st_is_set_piece": st_is_set_piece,
        "st_is_offensive_home": st_is_offensive_home,
        "st_is_offensive_away": st_is_offensive_away,
    }


def _empty_smart_tagging():
    return {
        "st_team": np.full(N_FRAMES, -1, dtype=np.int8),
        "st_is_set_piece": np.zeros(N_FRAMES, dtype=bool),
        "st_is_offensive_home": np.zeros(N_FRAMES, dtype=bool),
        "st_is_offensive_away": np.zeros(N_FRAMES, dtype=bool),
    }


# ---------------------------------------------------------------------------
# 4. Feature extraction
# ---------------------------------------------------------------------------

def extract_features(atd, st):
    """
    Build the feature matrix: one row per frame, many columns.

    All features are designed to be team-symmetric and handle NaN gracefully.
    """
    print("  Extracting features...")
    n = N_FRAMES
    feats = {}

    bx = atd["ball_x"]
    by = atd["ball_y"]
    px = atd["player_x"]  # (n, n_tracks)
    py = atd["player_y"]
    ps = atd["player_speed"]
    pt = atd["player_team"]  # (n_tracks,)

    home_mask = pt == 0
    away_mask = pt == 1

    # Convert ball to metres for distance calculations
    bx_m = bx * PITCH_LENGTH
    by_m = by * PITCH_WIDTH

    # ---- Ball features ----
    feats["ball_x"] = bx
    feats["ball_y"] = by
    feats["ball_visible"] = atd["ball_visible"].astype(np.float32)

    # Ball speed (from consecutive frames)
    ball_dx = np.diff(bx_m, prepend=np.nan)
    ball_dy = np.diff(by_m, prepend=np.nan)
    ball_speed = np.sqrt(ball_dx**2 + ball_dy**2) * FPS  # m/s
    feats["ball_speed"] = ball_speed

    # Ball acceleration
    ball_accel = np.diff(ball_speed, prepend=np.nan) * FPS
    feats["ball_accel"] = ball_accel

    # Ball direction (angle in radians)
    ball_dir = np.arctan2(ball_dy, ball_dx)
    feats["ball_direction"] = ball_dir

    # Ball distance from pitch centre
    feats["ball_dist_centre"] = np.sqrt(
        (bx_m - PITCH_LENGTH / 2)**2 + (by_m - PITCH_WIDTH / 2)**2
    )

    # ---- Per-team player-ball distances ----
    # Convert players to metres
    px_m = px * PITCH_LENGTH
    py_m = py * PITCH_WIDTH

    # Distance of each player to ball: (n, n_tracks)
    dist_to_ball = np.sqrt(
        (px_m - bx_m[:, None])**2 + (py_m - by_m[:, None])**2
    )

    # Home players
    home_dist = dist_to_ball[:, home_mask]  # (n, n_home_tracks)
    away_dist = dist_to_ball[:, away_mask]  # (n, n_away_tracks)

    # Nearest player to ball per team
    with np.errstate(all="ignore"):
        feats["nearest_home_dist"] = np.nanmin(home_dist, axis=1)
        feats["nearest_away_dist"] = np.nanmin(away_dist, axis=1)

    # Second nearest
    def second_min(arr):
        """Second smallest value per row, handling NaN."""
        out = np.full(arr.shape[0], np.nan, dtype=np.float32)
        for i in range(arr.shape[0]):
            vals = arr[i][np.isfinite(arr[i])]
            if len(vals) >= 2:
                vals.sort()
                out[i] = vals[1]
        return out

    feats["second_nearest_home_dist"] = second_min(home_dist)
    feats["second_nearest_away_dist"] = second_min(away_dist)

    # Ratio: who is closer?
    feats["dist_ratio"] = feats["nearest_home_dist"] / (
        feats["nearest_away_dist"] + 1e-6
    )

    # Count of players within 2m and 5m of ball
    for radius in [2.0, 5.0, 10.0]:
        tag = f"_{int(radius)}m"
        feats[f"home_count{tag}"] = np.nansum(home_dist < radius, axis=1).astype(np.float32)
        feats[f"away_count{tag}"] = np.nansum(away_dist < radius, axis=1).astype(np.float32)
        feats[f"total_count{tag}"] = feats[f"home_count{tag}"] + feats[f"away_count{tag}"]

    # ---- Team formation features ----
    # Centroids
    home_px = px_m[:, home_mask]
    home_py = py_m[:, home_mask]
    away_px = px_m[:, away_mask]
    away_py = py_m[:, away_mask]

    with np.errstate(all="ignore"):
        feats["home_centroid_x"] = np.nanmean(home_px, axis=1)
        feats["home_centroid_y"] = np.nanmean(home_py, axis=1)
        feats["away_centroid_x"] = np.nanmean(away_px, axis=1)
        feats["away_centroid_y"] = np.nanmean(away_py, axis=1)

    # Team spread (std of positions)
    feats["home_spread_x"] = np.nanstd(home_px, axis=1)
    feats["home_spread_y"] = np.nanstd(home_py, axis=1)
    feats["away_spread_x"] = np.nanstd(away_px, axis=1)
    feats["away_spread_y"] = np.nanstd(away_py, axis=1)

    # Ball distance from each team's centroid
    feats["ball_to_home_centroid"] = np.sqrt(
        (bx_m - feats["home_centroid_x"])**2 +
        (by_m - feats["home_centroid_y"])**2
    )
    feats["ball_to_away_centroid"] = np.sqrt(
        (bx_m - feats["away_centroid_x"])**2 +
        (by_m - feats["away_centroid_y"])**2
    )

    # Centroid ratio
    feats["centroid_dist_ratio"] = feats["ball_to_home_centroid"] / (
        feats["ball_to_away_centroid"] + 1e-6
    )

    # ---- Player count and tracking quality ----
    home_visible = np.sum(np.isfinite(home_px), axis=1).astype(np.float32)
    away_visible = np.sum(np.isfinite(away_px), axis=1).astype(np.float32)
    feats["home_players_visible"] = home_visible
    feats["away_players_visible"] = away_visible
    feats["total_players_visible"] = home_visible + away_visible

    # ---- Average team speed ----
    home_speeds = ps[:, home_mask]
    away_speeds = ps[:, away_mask]
    feats["home_avg_speed"] = np.nanmean(home_speeds, axis=1)
    feats["away_avg_speed"] = np.nanmean(away_speeds, axis=1)
    feats["home_max_speed"] = np.nanmax(home_speeds, axis=1)
    feats["away_max_speed"] = np.nanmax(away_speeds, axis=1)

    # ---- Ball-missing gap length ----
    # How many consecutive frames the ball has been missing
    ball_vis = atd["ball_visible"]
    gap_len = np.zeros(n, dtype=np.float32)
    current_gap = 0
    for i in range(n):
        if ball_vis[i]:
            current_gap = 0
        else:
            current_gap += 1
        gap_len[i] = current_gap
    feats["ball_gap_length"] = gap_len

    # Time since ball was last seen (seconds)
    feats["ball_gap_seconds"] = gap_len / FPS

    # ---- Smart Tagging features ----
    feats["st_team"] = st["st_team"].astype(np.float32)
    feats["st_is_set_piece"] = st["st_is_set_piece"].astype(np.float32)
    feats["st_is_offensive_home"] = st["st_is_offensive_home"].astype(np.float32)
    feats["st_is_offensive_away"] = st["st_is_offensive_away"].astype(np.float32)

    # Smart tagging team agreement with nearest-player heuristic
    nearest_team = np.where(
        feats["nearest_home_dist"] < feats["nearest_away_dist"], 0, 1
    ).astype(np.float32)
    nearest_team[~ball_vis] = np.nan
    feats["st_agrees_nearest"] = (
        st["st_team"].astype(np.float32) == nearest_team
    ).astype(np.float32)

    # ---- Half indicator ----
    half_arr = np.zeros(n, dtype=np.float32)
    for half_num, start_f, end_f in HALF_BLOCKS:
        half_arr[start_f:end_f + 1] = half_num
    feats["half"] = half_arr

    # ---- Temporal / rolling features ----
    # Rolling window stats over key signals
    print("  Computing rolling features...")
    for window_frames in [25, 75]:  # 1s, 3s
        suffix = f"_roll{window_frames}"
        for key in ["nearest_home_dist", "nearest_away_dist", "ball_speed",
                     "dist_ratio", "ball_visible"]:
            arr = feats[key]
            series = pd.Series(arr)
            feats[f"{key}{suffix}_mean"] = (
                series.rolling(window_frames, center=True, min_periods=1)
                .mean().values.astype(np.float32)
            )

    # Build DataFrame
    df = pd.DataFrame(feats)
    df.index.name = "video_frame"

    print(f"  Feature matrix: {df.shape[0]} frames × {df.shape[1]} features")
    return df


# ---------------------------------------------------------------------------
# 5. Main: build and save feature matrix
# ---------------------------------------------------------------------------

def build_features(save_path=None):
    """Load data, extract features, optionally save to parquet/csv."""
    paths = asset_paths()

    print("[1/3] Loading ATD tracking...")
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team_of_track = load_team_assignments(meta_path)
    atd = parse_atd_raw(raw_path, team_of_track, N_FRAMES)

    print("\n[2/3] Loading Smart Tagging...")
    st_path = paths.get("smart_tagging")
    if st_path and os.path.isfile(st_path):
        st = parse_smart_tagging(st_path)
    else:
        print("  Smart Tagging not found, using empty priors")
        st = _empty_smart_tagging()

    print("\n[3/3] Extracting features...")
    df = extract_features(atd, st)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_parquet(save_path)
        print(f"\nSaved features to {save_path}")

    return df


if __name__ == "__main__":
    save_path = os.path.join(ROOT, "pipeline", "output", "features.parquet")
    build_features(save_path)

