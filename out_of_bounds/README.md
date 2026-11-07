
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
```
