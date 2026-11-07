
from out_of_bounds.detector import calculate_ball_velocity

from out_of_bounds.event_features import (
    analyze_outside_events,
    classify_exit_evidence,
    validate_boundary_crossings,
    find_full_ball_exit_frames,
    add_exit_reliability,
    validate_full_exit_continuity,
)


def detect_out_of_bounds(tracking, half_ranges=None):
    """
    Detect out-of-bounds events from ball tracking.
    Returns event candidates, not final D labels.
    """

    ball = tracking[
        ["video_frame", "ball_x", "ball_y"]
    ].copy()

    ball = ball.sort_values("video_frame")

    # Step 1: Calculate ball velocity
    ball = calculate_ball_velocity(ball)

    # Step 2: Extract outside events
    events = analyze_outside_events(ball)

    # Step 3: Restrict to gameplay
    if half_ranges is not None:
        mask = False

        for start, end in half_ranges:
            mask = mask | events["start_frame"].between(
                start, end
            )

        events = events.loc[mask].copy()

    # Step 4: Classify geometric evidence
    events = classify_exit_evidence(events)

    # Step 5: Validate inside-to-outside transitions
    events = validate_boundary_crossings(ball, events)

    # Step 6: Find whole-ball exit frames
    events = find_full_ball_exit_frames(ball, events)

    # Step 7: Assign preliminary reliability
    events = add_exit_reliability(events)

    # Step 8: Check tracking continuity
    events = validate_full_exit_continuity(ball, events)

    return events.reset_index(drop=True)
