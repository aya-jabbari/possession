
from kloppy import metrica

from games import asset_paths
from out_of_bounds.detector import detect_kick_candidates

paths = asset_paths()

dataset = metrica.load_tracking_epts(
    meta_data=paths["atd_meta"],
    raw_data=paths["atd_raw"],
    limit=18685
)

df = dataset.to_df()
df["video_frame"] = df["frame_id"] - 1

# Include every player track and ball
window = df.loc[
    df["video_frame"].between(18250, 18550)
].copy()

candidates = detect_kick_candidates(window)

print("===== KICK CANDIDATES =====")

for candidate in candidates:
    print(candidate)

print("\nTotal candidates:", len(candidates))

from out_of_bounds.detector import group_kick_candidates

kick_events = group_kick_candidates(candidates)

print("\n===== GROUPED KICK EVENTS =====")

for event in kick_events:
    print(event)

print("\nTotal grouped events:", len(kick_events))
