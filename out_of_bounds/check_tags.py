
from kloppy import sportscode
from games import asset_paths

paths = asset_paths()

events = sportscode.load(
    paths["smart_tagging"]
).to_df()

# Smart Tagging timestamps are on the tactical video timeline
events["start_frame"] = (
    events["timestamp"].dt.total_seconds() * 25
).round().astype(int)

events["end_frame"] = (
    events["end_timestamp"].dt.total_seconds() * 25
).round().astype(int)

# Events overlapping the region around Event 2
nearby = events.loc[
    (events["start_frame"] <= 19000) &
    (events["end_frame"] >= 17500)
].copy()

columns = [
    c for c in
    ["start_frame", "end_frame", "code", "Type", "Team"]
    if c in nearby.columns
]

print("\n===== SMART TAGGING NEAR EVENT 2 =====")
print(nearby[columns].to_string(index=False))
