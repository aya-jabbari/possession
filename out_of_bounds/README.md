
# Out-of-Bounds Detection Module

## Purpose

Detect candidate ball exits from the football pitch
using tracking coordinates and ball motion.

## Input

A pandas DataFrame containing:

- video_frame
- ball_x
- ball_y

Coordinates are normalized to the pitch.

## Usage

```python
from out_of_bounds.pipeline import detect_out_of_bounds

events = detect_out_of_bounds(
    tracking,
    half_ranges=[
        (11185, 90651),
        (110808, 190197)
    ]
)
```

## Output

One row per candidate event.

Important fields:

- full_ball_exit_frame
- boundary
- outward_speed_mps
- crossing_status
- exit_reliability
- full_exit_continuous

## Reliability categories

- observed_full_exit:
  Continuous tracking supports a complete boundary crossing.
- exit_time_unknown:
  Ball observed outside after missing tracking.
- no_full_crossing:
  No observation of the whole ball outside.

## Limitations

This module does not predict possession or referee stoppages.
It does not determine when play resumes.
Event coordinates can contain tracking errors.

Do not convert all outside-event frames directly to D labels.

## Model handoff

The generated files are automatic intermediate signals, not perfect labels:

- `events.csv`: candidate ball-exit events.
- `kick_candidates.csv`: possible restart/kick events.
- `tentative_dead_ball_intervals.csv`: conservative candidate dead-ball intervals.
- `frame_features.csv`: one row per tactical `video_frame`, aligned to the model's frame index.

`frame_features.csv` contains binary event indicators, exit boundary and reliability
fields, plus frame distances to the nearest exit and restart. A zero means that this
detector found no evidence for that feature; it does not mean the ball was definitely
in play. A one means candidate evidence, not a verified ground-truth label.

Merge the features with the possession model's frame-level table using an integer
`video_frame` key. Keep the tracking/model table as the primary table and use a
left join so every match frame is retained:

```python
features = pd.read_csv("out_of_bounds/output/frame_features.csv")
model_frames = model_frames.merge(features, on="video_frame", how="left")
```

After the merge, fill only genuinely missing feature values with the neutral defaults
used by the model (usually `0` for binary indicators and `-1` for distance fields).
Do not treat `dead_ball_candidate` as the final `D` class. Combine it with tracking,
Smart Tagging, and other model features, then apply the model's own temporal
post-processing to produce the final `H`, `A`, `D`, or `X` submission.
```
