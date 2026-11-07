
from kloppy import metrica
from games import asset_paths

paths = asset_paths()

# Load enough frames to cover the first 5 min of play
dataset = metrica.load_tracking_epts(
    meta_data=paths["atd_meta"],
    raw_data=paths["atd_raw"],
    limit=18685
)

df = dataset.to_df()

# ATD frame_id starts at 1
# Video frame starts at 0
df["video_frame"] = df["frame_id"] - 1

# Only analyze actual gameplay
ball = df.loc[
    (df["video_frame"] >= 11185) &
    (df["video_frame"] <= 18684),
    ["video_frame", "ball_x", "ball_y"]
].copy()

# Check available positions
valid = ball[["ball_x", "ball_y"]].notna().all(axis=1)

# Detect outside positions
outside = valid & (
    (ball["ball_x"] < 0) |
    (ball["ball_x"] > 1) |
    (ball["ball_y"] < 0) |
    (ball["ball_y"] > 1)
)

print("Total gameplay frames:", len(ball))
print("Valid ball positions:", valid.sum())
print("Missing ball positions:", (~valid).sum())
print("Out-of-bounds candidates:", outside.sum())

print("\nFirst outside observations:")
print(ball.loc[outside].head(20))

print("\nCoordinate ranges:")
print("X:", ball["ball_x"].min(), ball["ball_x"].max())
print("Y:", ball["ball_y"].min(), ball["ball_y"].max())


from out_of_bounds.detector import group_outside_events

events = group_outside_events(ball)

print("\nDETECTED OUT-OF-BOUNDS EVENTS:")
print(events.to_string(index=False))

print("\nTotal events:", len(events))

if not events.empty:
    longest = events.loc[events["duration_frames"].idxmax()]
    print("\nLongest outside event:")
    print(longest)


# Investigate Event 1
segment = ball.loc[
    (ball["video_frame"] >= 17450) &
    (ball["video_frame"] <= 17560)
].copy()

segment["visible"] = (
    segment["ball_x"].notna() &
    segment["ball_y"].notna()
)

print("\nEVENT 1 - AFTER EXIT")
print(segment[
    ["video_frame", "ball_x", "ball_y", "visible"]
].to_string(index=False))


# Find the first visible ball after Event 1
after_exit = ball.loc[
    ball["video_frame"] > 17459
].copy()

visible_after = after_exit.loc[
    after_exit["ball_x"].notna()
    & after_exit["ball_y"].notna()
]

if not visible_after.empty:
    first = visible_after.iloc[0]

    return_frame = int(first["video_frame"])
    missing_frames = return_frame - 17459 - 1

    print("\nFIRST BALL DETECTION AFTER EVENT 1")
    print("Frame:", return_frame)
    print("Missing frames:", missing_frames)
    print("Missing duration:", round(missing_frames / 25, 2), "s")
    print("Coordinates:", first["ball_x"], first["ball_y"])
else:
    print("Ball not detected again within loaded frames.")



from out_of_bounds.detector import calculate_ball_velocity

# Calculate velocity for all loaded frames
ball_with_speed = calculate_ball_velocity(ball)

# Focus on Event 1: frames 17450–17459
event1 = ball_with_speed.loc[
    ball_with_speed["video_frame"].between(17450, 17459),
    ["video_frame", "ball_x", "ball_y", "vx", "vy", "speed"]
]

print("\n===== EVENT 1 VELOCITY TEST =====")
print(event1.to_string(index=False))

# Check if the ball was moving outward
exit_frame = ball_with_speed.loc[
    ball_with_speed["video_frame"] == 17456
]

if not exit_frame.empty:
    vx = exit_frame.iloc[0]["vx"]
    speed = exit_frame.iloc[0]["speed"]

    print("\nExit frame: 17456")
    print(f"Horizontal velocity: {vx:.2f} m/s")
    print(f"Total speed: {speed:.2f} m/s")

    if vx < 0:
        print("RESULT: Ball moving OUTWARD across left goal line")
    else:
        print("RESULT: Ball NOT moving outward")


# Analyze Event 2 in detail
event2 = ball_with_speed.loc[
    ball_with_speed["video_frame"].between(18020, 18135),
    ["video_frame", "ball_x", "ball_y", "vx", "vy", "speed"]
]

print("\n===== EVENT 2 - FIRST 15 FRAMES =====")
print(event2.head(15).to_string(index=False))

print("\n===== EVENT 2 - LAST 15 FRAMES =====")
print(event2.tail(15).to_string(index=False))

# Inspect the first visible position after Event 2
after_event2 = ball_with_speed.loc[
    ball_with_speed["video_frame"] > 18126
].dropna(subset=["ball_x", "ball_y"])

if not after_event2.empty:
    first = after_event2.iloc[0]
    frame = int(first["video_frame"])

    print("\nFIRST DETECTION AFTER EVENT 2")
    print("Frame:", frame)
    print("Missing frames:", frame - 18126 - 1)
    print("Coordinates:", first["ball_x"], first["ball_y"])


# Inspect the full trajectory around Event 2
event2 = ball_with_speed.loc[
    ball_with_speed["video_frame"].between(18020, 18350)
].copy()

# Find the first transition from negative to positive vx
reversals = event2.loc[
    (event2["vx"] > 0) &
    (event2["vx"].shift(1) <= 0)
]

print("\n=== VELOCITY REVERSAL ===")
if not reversals.empty:
    reversal_frame = int(reversals.iloc[0]["video_frame"])
    print("Reversal frame:", reversal_frame)

    print(event2.loc[
        event2["video_frame"].between(
            reversal_frame - 5,
            reversal_frame + 5
        ),
        ["video_frame", "ball_x", "ball_y",
         "vx", "vy", "speed"]
    ].to_string(index=False))
else:
    print("No observed negative-to-positive vx reversal")

print("\n=== WHAT HAPPENED AFTER FRAME 18135? ===")

# Sample every 10 frames
after = event2.loc[
    event2["video_frame"].between(18135, 18350)
]

print(
    after.iloc[::10][
        ["video_frame", "ball_x", "ball_y",
         "vx", "vy", "speed"]
    ].to_string(index=False)
)
