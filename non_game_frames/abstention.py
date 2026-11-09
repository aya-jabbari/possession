"""Task 4: abstention (X) on non-game frames.

Non-game frames (pre-match, halftime, post-match) are never scored, so
labelling them X costs nothing. Inside the match X is NOT free, so it must
appear only outside the two halves.

Usage (call LAST in the pipeline, after all other tasks):

    from non_game_frames.abstention import apply_abstention
    final_labels = apply_abstention(labels)
"""
import numpy as np

N_FRAMES = 199_319  # one row per video frame: 0 .. 199318

# Half boundaries in video frames, both ends inclusive.
# test_abstention.py cross-checks these against games.py.
HALF_BLOCKS = [
    (11_185, 90_651),    # first half
    (110_808, 190_197),  # second half
]


def non_game_mask(n_frames: int = N_FRAMES) -> np.ndarray:
    """Boolean array, True where the frame is non-game (label must be X)."""
    in_play = np.zeros(n_frames, dtype=bool)
    for first, last in HALF_BLOCKS:
        in_play[first:last + 1] = True
    return ~in_play


def apply_abstention(labels) -> np.ndarray:
    """Return a copy of `labels` (H/A/D) with X on every non-game frame."""
    out = np.asarray(labels, dtype=object).copy()
    out[non_game_mask(len(out))] = "X"
    return out