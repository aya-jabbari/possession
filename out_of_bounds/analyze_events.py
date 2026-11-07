
from kloppy import metrica

from games import asset_paths
from out_of_bounds.detector import calculate_ball_velocity

paths = asset_paths()

dataset = metrica.load_tracking_epts(
    meta_data=paths["atd_meta"],
    raw_data=paths["atd_raw"],
    limit=18685
)

df = dataset.to_df()
df["video_frame"] = df["frame_id"] - 1

ball = df[
    ["video_frame", "ball_x", "ball_y"]
].copy()

ball = calculate_ball_velocity(ball)

# Approximate normalized radius
radius_x = 0.11 / 105.3

# Analyze Event 2
event = ball.loc[
    ball["video_frame"].between(18015, 18140)
].copy()

# Whole-ball crossing of left goal line
fully_outside = event["ball_x"] < -radius_x

# Outward/inward motion
event["moving_outward"] = event["vx"] < 0
event["moving_inward"] = event["vx"] > 0

outside = event.loc[fully_outside]

print("===== EVENT 2 DIAGNOSTICS =====")

if not outside.empty:
    print("First whole-ball outside frame:",
          int(outside["video_frame"].iloc[0]))

    print("Maximum distance outside (m):",
          round(-outside["ball_x"].min() * 105.3, 2))

# Detect outward-to-inward velocity reversal
reversal = event.loc[
    (event["vx"] > 0) &
    (event["vx"].shift(1) <= 0)
]

if not reversal.empty:
    print("First velocity reversal frame:",
          int(reversal["video_frame"].iloc[0]))

# Find return after observed exit
if not outside.empty:
    first_out = int(outside["video_frame"].iloc[0])

    returned = event.loc[
        (event["video_frame"] > first_out) &
        (event["ball_x"] >= radius_x)
    ]

    if not returned.empty:
        print("Whole ball back inside at frame:",
              int(returned["video_frame"].iloc[0]))
    else:
        print("No full return observed in this window")

print("\n===== EVENT 2 - AFTER RETURN =====")

after_return = ball.loc[
    ball["video_frame"].between(18128, 18650)
].copy()

# Sample every 25 frames = 1 second
sample = after_return.iloc[::25]

print(sample[
    ["video_frame", "ball_x", "ball_y",
     "vx", "vy", "speed"]
].to_string(index=False))

print("\n===== TRACKING COVERAGE =====")

print("Frames:", len(after_return))
print(
    "Visible:",
    after_return[["ball_x", "ball_y"]]
    .notna().all(axis=1).sum()
)

print(
    "Missing:",
    after_return[["ball_x", "ball_y"]]
    .isna().any(axis=1).sum()
)

print("\n===== GOAL KICK RESTART INVESTIGATION =====")

restart_window = ball.loc[
    ball["video_frame"].between(18250, 18550)
].copy()

# A simple acceleration estimate
restart_window["acceleration"] = (
    restart_window["speed"].diff()
    / (restart_window["video_frame"].diff() / 25)
)

# Summarize every 5 frames
sample = restart_window.iloc[::5]

print(sample[
    ["video_frame", "ball_x", "ball_y",
     "speed", "acceleration"]
].to_string(index=False))

print("\nMaximum speed increase per second:")
print(
    restart_window[
        ["video_frame", "acceleration"]
    ].nlargest(5, "acceleration")
)

import numpy as np

print("\n===== POSSIBLE BALL KICKS =====")

# Use the original tracking DataFrame (df), which also
# contains the player-position columns.
player_x_cols = [
    c for c in df.columns
    if c.endswith("_x") and c != "ball_x"
]

for frame in [18345, 18355, 18375, 18415, 18430, 18445]:
    row = df.loc[df["video_frame"] == frame]

    if row.empty:
        continue

    row = row.iloc[0]
    bx, by = row["ball_x"], row["ball_y"]

    distances = []

    for x_col in player_x_cols:
        y_col = x_col[:-2] + "_y"

        if y_col not in df.columns:
            continue

        px, py = row[x_col], row[y_col]

        if not np.isfinite([bx, by, px, py]).all():
            continue

        distance = np.hypot(
            (px - bx) * 105.3,
            (py - by) * 68
        )

        distances.append((x_col[:-2], distance))

    distances.sort(key=lambda item: item[1])

    print(f"\nFrame {frame}")
    print("Ball speed:",
          round(float(ball.loc[
              ball["video_frame"] == frame,
              "speed"
          ].iloc[0]), 2), "m/s")
    print("Closest player tracks:", distances[:3])
