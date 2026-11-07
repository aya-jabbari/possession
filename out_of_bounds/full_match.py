
from kloppy import metrica

from games import asset_paths, source_frames
from out_of_bounds.detector import (
    calculate_ball_velocity,
    group_outside_events
)

paths = asset_paths()

print("Loading full-match tracking...")

dataset = metrica.load_tracking_epts(
    meta_data=paths["atd_meta"],
    raw_data=paths["atd_raw"]
)

df = dataset.to_df()

df["video_frame"] = df["frame_id"] - 1

from out_of_bounds.pipeline import detect_out_of_bounds

half_ranges = [
    (11185, 90651),
    (110808, 190197),
]

api_events = detect_out_of_bounds(
    tracking=df,
    half_ranges=half_ranges
)

print("\n===== INTEGRATION API TEST =====")
print("Events:", len(api_events))

print(
    api_events["exit_reliability"].value_counts()
)

print("\nColumns:")
print(api_events.columns.tolist())

assert len(api_events) == 81
assert (
    api_events["exit_reliability"]
    .value_counts()
    .get("observed_full_exit", 0)
    == 50
)

print("\nAPI test passed!")


print("Tracking frames:", len(df))
print("Expected frames:", source_frames())

ball = df[
    ["video_frame", "ball_x", "ball_y"]
].copy()

ball = calculate_ball_velocity(ball)

events = group_outside_events(ball)

print("\n===== FULL-MATCH OUTSIDE EVENTS =====")
print("Total outside events:", len(events))
print(events.head(20).to_string(index=False))

print("\n===== SUMMARY =====")
print("Observed ball frames:", ball["ball_x"].notna().sum())
print("Missing ball frames:", ball["ball_x"].isna().sum())

# Save only derived event metadata.
# Do not publish the competition's private tracking.

from out_of_bounds.event_features import analyze_outside_events

print("\n===== EVENT FEATURES =====")

features = analyze_outside_events(ball)

print(features.head(20).to_string(index=False))

print("\n===== BOUNDARY DISTRIBUTION =====")
print(features["boundary"].value_counts())

print("\n===== FULL BALL OUTSIDE =====")
print(features["fully_outside_observed"].value_counts())

from out_of_bounds.event_features import classify_exit_evidence
from games import half_video_blocks

# Only keep events starting in a graded half
halves = half_video_blocks("BAR-ATH")

in_play = False

for half, start, end in halves:
    mask = features["start_frame"].between(start, end)

    if isinstance(in_play, bool):
        in_play = mask
    else:
        in_play = in_play | mask

gameplay_events = features.loc[in_play].copy()

gameplay_events = classify_exit_evidence(
    gameplay_events
)

print("\n===== GAMEPLAY EVENT EVIDENCE =====")

print(
    gameplay_events[
        ["start_frame", "end_frame", "boundary",
         "outward_speed_mps", "max_outside_distance_m",
         "evidence_level"]
    ].head(25).to_string(index=False)
)

print("\n===== EVIDENCE DISTRIBUTION =====")

print(
    gameplay_events["evidence_level"].value_counts()
)

from out_of_bounds.event_features import validate_boundary_crossings

gameplay_events = validate_boundary_crossings(
    ball,
    gameplay_events
)

print("\n===== CROSSING STATUS =====")
print(
    gameplay_events["crossing_status"].value_counts()
)

print("\n===== CROSSING DETAILS =====")
print(
    gameplay_events[
        ["start_frame", "boundary",
         "evidence_level", "crossing_status"]
    ].head(25).to_string(index=False)
)

from out_of_bounds.event_features import find_full_ball_exit_frames

gameplay_events = find_full_ball_exit_frames(
    ball,
    gameplay_events
)

print("\n===== FULL BALL EXIT FRAMES =====")

print(
    gameplay_events[
        ["start_frame", "full_ball_exit_frame",
         "boundary", "crossing_status",
         "evidence_level"]
    ].head(25).to_string(index=False)
)

print("\n===== EXIT FRAME COVERAGE =====")

print(
    "With full-ball observation:",
    gameplay_events["full_ball_exit_frame"].notna().sum()
)

print(
    "Without full-ball observation:",
    gameplay_events["full_ball_exit_frame"].isna().sum()
)

from out_of_bounds.event_features import add_exit_reliability

gameplay_events = add_exit_reliability(gameplay_events)

print("\n===== FINAL EXIT RELIABILITY =====")

print(
    gameplay_events["exit_reliability"].value_counts()
)

print("\n===== SAMPLE EVENTS =====")

print(
    gameplay_events[
        ["start_frame", "full_ball_exit_frame",
         "boundary", "crossing_status",
         "exit_reliability"]
    ].head(25).to_string(index=False)
)

print("\n===== CONTINUITY VALIDATION =====")

print(
    api_events["full_exit_continuous"].value_counts()
)

print("\n===== UPDATED RELIABILITY =====")

print(
    api_events["exit_reliability"].value_counts()
)

print("\n===== UNVERIFIED CONTINUITY =====")

print(
    api_events.loc[
        api_events["exit_reliability"] == "continuity_unverified",
        ["start_frame", "full_ball_exit_frame",
         "boundary", "crossing_status"]
    ].to_string(index=False)
)
