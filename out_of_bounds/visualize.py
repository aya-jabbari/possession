
import matplotlib.pyplot as plt
from kloppy import metrica

from games import asset_paths
from out_of_bounds.detector import group_outside_events

paths = asset_paths()

dataset = metrica.load_tracking_epts(
    meta_data=paths["atd_meta"],
    raw_data=paths["atd_raw"],
    limit=18685
)

df = dataset.to_df()
df["video_frame"] = df["frame_id"] - 1

ball = df.loc[
    (df["video_frame"] >= 11185)
    & (df["video_frame"] <= 18684),
    ["video_frame", "ball_x", "ball_y"]
].copy()

events = group_outside_events(ball)

for i, event in events.iterrows():

    start = int(event["start_frame"])
    end = int(event["end_frame"])

    # Inspect 50 frames before and after
    segment = ball.loc[
        (ball["video_frame"] >= start - 50)
        & (ball["video_frame"] <= end + 50)
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                            sharex=True)

    axes[0].plot(
        segment["video_frame"],
        segment["ball_x"],
        label="Ball X"
    )

    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].axhline(1, color="red", linestyle="--")
    axes[0].set_ylabel("Normalized X")
    axes[0].legend()

    axes[1].plot(
        segment["video_frame"],
        segment["ball_y"],
        label="Ball Y"
    )

    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].axhline(1, color="red", linestyle="--")
    axes[1].set_ylabel("Normalized Y")
    axes[1].set_xlabel("Video frame")
    axes[1].legend()

    for ax in axes:
        ax.axvspan(start, end, alpha=0.2)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Out-of-bounds candidate {i + 1}: "
        f"frames {start}-{end}"
    )

    plt.tight_layout()
    plt.show()

# Inspect ball availability around Event 1
segment = ball.loc[
    (ball["video_frame"] >= 17450) &
    (ball["video_frame"] <= 17560)
].copy()

segment["ball_visible"] = (
    segment["ball_x"].notna() &
    segment["ball_y"].notna()
)

print("\nEVENT 1 - AFTER BOUNDARY CROSSING")
print(segment[
    ["video_frame", "ball_x", "ball_y", "ball_visible"]
].to_string(index=False))
