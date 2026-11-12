"""
Step 3: Viterbi Decoder
=======================
HMM-based temporal decoding of GBDT emission probabilities.

The Hidden Markov Model has 3 states: H (home), A (away), D (dead ball).
- Emission probabilities come from the GBDT model (Step 2)
- Transition probabilities encode football domain knowledge:
  - Possessions are "sticky" (high self-transition probability)
  - H↔A transitions are rare (usually goes through D)
  - D→H and D→A are restart events

The Viterbi algorithm finds the most likely STATE SEQUENCE given all
observations. It encourages temporally coherent possessions; it does not
directly optimise the evaluation metric spellF1.

Usage:
    python -m pipeline.viterbi
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks  # noqa: E402

N_FRAMES = GAMES[GAME]["n_video_frames"]
HALF_BLOCKS = half_video_blocks(GAME)  # [(1, 11185, 90651), (2, 110808, 190197)]

# States
H, A, D = 0, 1, 2
STATE_NAMES = {0: "H", 1: "A", 2: "D"}


# ---------------------------------------------------------------------------
# 1. Transition matrix — football domain knowledge
# ---------------------------------------------------------------------------

def build_transition_matrix(
    stay_possession=0.998,
    possession_to_dead=0.001,
    possession_to_other=0.001,
    stay_dead=0.990,
    dead_to_home=0.005,
    dead_to_away=0.005,
):
    """
    Build the 3×3 log-transition matrix for the HMM.

    Key design decisions:
    - stay_possession: how "sticky" a possession is. Higher = fewer
      transitions = fewer predicted possessions. This is the MAIN
      lever for spellF1.
    - possession_to_dead: cost of going from live play to dead ball.
      This captures fouls, out-of-bounds, etc.
    - possession_to_other: cost of direct H→A or A→H transition
      (without going through D). Should be low — most possession
      changes go through a dead ball, but some are direct interceptions.
    - stay_dead: how sticky dead ball is. Set pieces can take a while.
    - dead_to_home / dead_to_away: restart probabilities.

    Returns:
        log_trans: (3, 3) array of log-probabilities
    """
    trans = np.array([
        # To:    H                    A                       D
        [stay_possession,    possession_to_other,    possession_to_dead],   # From H
        [possession_to_other, stay_possession,       possession_to_dead],   # From A
        [dead_to_home,       dead_to_away,           stay_dead],            # From D
    ], dtype=np.float64)

    # Normalise rows to sum to 1
    trans = trans / trans.sum(axis=1, keepdims=True)

    print("Transition matrix:")
    print(f"  P(stay in possession) = {trans[0, 0]:.6f}")
    print(f"  P(possession → other) = {trans[0, 1]:.6f}")
    print(f"  P(possession → dead)  = {trans[0, 2]:.6f}")
    print(f"  P(stay dead)          = {trans[2, 2]:.6f}")
    print(f"  P(dead → restart)     = {trans[2, 0]:.6f}")

    return np.log(trans + 1e-300)


# ---------------------------------------------------------------------------
# 2. Viterbi algorithm (log-space, numerically stable)
# ---------------------------------------------------------------------------

def viterbi(emissions, log_trans, log_prior=None):
    """
    Viterbi algorithm in log-space.

    Args:
        emissions: (T, 3) array of P(observation | state) from GBDT
        log_trans: (3, 3) array of log-transition probabilities
        log_prior: (3,) array of log-prior probabilities for t=0.
                   If None, uses uniform.

    Returns:
        path: (T,) array of optimal state indices (0=H, 1=A, 2=D)
    """
    T, K = emissions.shape
    assert K == 3, f"Expected 3 states, got {K}"

    # Clamp emissions to avoid log(0)
    log_emit = np.log(np.clip(emissions, 1e-10, 1.0))

    # Initialise
    if log_prior is None:
        log_prior = np.log(np.ones(K) / K)

    # Viterbi tables
    V = np.zeros((T, K), dtype=np.float64)     # max log-prob
    B = np.zeros((T, K), dtype=np.int32)        # backpointer

    V[0] = log_prior + log_emit[0]

    # Forward pass
    for t in range(1, T):
        for j in range(K):
            # V[t-1, i] + log_trans[i, j] for all i
            scores = V[t - 1] + log_trans[:, j]
            B[t, j] = np.argmax(scores)
            V[t, j] = scores[B[t, j]] + log_emit[t, j]

    # Backward pass (traceback)
    path = np.zeros(T, dtype=np.int32)
    path[-1] = np.argmax(V[-1])

    for t in range(T - 2, -1, -1):
        path[t] = B[t + 1, path[t + 1]]

    return path


def viterbi_vectorised(emissions, log_trans, log_prior=None):
    """
    Vectorised Viterbi (faster for long sequences).
    Same interface as viterbi() but uses numpy broadcasting.
    """
    T, K = emissions.shape

    log_emit = np.log(np.clip(emissions, 1e-10, 1.0))

    if log_prior is None:
        log_prior = np.log(np.ones(K) / K)

    V = np.zeros((T, K), dtype=np.float64)
    B = np.zeros((T, K), dtype=np.int32)

    V[0] = log_prior + log_emit[0]

    for t in range(1, T):
        # (K, K): score[i, j] = V[t-1, i] + log_trans[i, j]
        scores = V[t - 1, :, None] + log_trans  # (K, K)
        B[t] = np.argmax(scores, axis=0)         # best predecessor for each state
        V[t] = scores[B[t], np.arange(K)] + log_emit[t]

    # Traceback
    path = np.zeros(T, dtype=np.int32)
    path[-1] = np.argmax(V[-1])
    for t in range(T - 2, -1, -1):
        path[t] = B[t + 1, path[t + 1]]

    return path


# ---------------------------------------------------------------------------
# 3. Decode each half independently
# ---------------------------------------------------------------------------

def decode_match(emissions, log_trans, initial_states=None):
    """
    Run Viterbi on each half separately.

    The match has two halves with a break between them. Each half starts
    fresh (the state at the end of half 1 doesn't carry to half 2).

    Args:
        emissions: (N_FRAMES, 3) emission probabilities
        log_trans: (3, 3) log-transition matrix
        initial_states: dict {half_num: initial_state_int} or None

    Returns:
        labels: (N_FRAMES,) array of state indices
    """
    labels = np.full(N_FRAMES, D, dtype=np.int32)  # default to D outside halves

    for half_num, start_f, end_f in HALF_BLOCKS:
        half_emissions = emissions[start_f:end_f + 1]
        n_half = len(half_emissions)

        # Set prior: if initial state is known, bias heavily
        if initial_states and half_num in initial_states:
            log_prior = np.full(3, -10.0)
            log_prior[initial_states[half_num]] = 0.0
        else:
            # Uniform prior
            log_prior = np.log(np.ones(3) / 3)

        print(f"  Half {half_num}: decoding {n_half} frames "
              f"({start_f}..{end_f})...")

        path = viterbi_vectorised(half_emissions, log_trans, log_prior)
        labels[start_f:end_f + 1] = path

        # Stats
        for s in [H, A, D]:
            count = np.sum(path == s)
            print(f"    {STATE_NAMES[s]}: {count} frames ({100 * count / n_half:.1f}%)")

    return labels


# ---------------------------------------------------------------------------
# 4. Post-processing
# ---------------------------------------------------------------------------

def remove_short_segments(labels, min_frames=5):
    """
    Remove possession segments shorter than min_frames.

    Very short segments (< 0.2s) are almost certainly noise. Replace them
    with the surrounding label.

    This is a secondary smoothing step — Viterbi should handle most of it,
    but this catches edge cases.
    """
    result = labels.copy()
    n = len(result)

    i = 0
    while i < n:
        # Find the end of this segment
        j = i + 1
        while j < n and result[j] == result[i]:
            j += 1

        seg_len = j - i
        seg_label = result[i]

        # Only clean up H/A segments (not D — dead balls can be short)
        if seg_len < min_frames and seg_label in (H, A):
            # Replace with the label before this segment
            if i > 0:
                result[i:j] = result[i - 1]
            elif j < n:
                result[i:j] = result[j]

        i = j

    return result


def labels_to_submission(labels):
    """Convert integer labels to submission-format string labels."""
    label_strings = np.full(N_FRAMES, "X", dtype="<U1")

    for half_num, start_f, end_f in HALF_BLOCKS:
        for i in range(start_f, end_f + 1):
            label_strings[i] = STATE_NAMES.get(labels[i], "X")

    return label_strings


# ---------------------------------------------------------------------------
# 5. Write submission
# ---------------------------------------------------------------------------

def write_submission(labels_str, path):
    """Write a submission CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("frame,label\n")
        for i, label in enumerate(labels_str):
            f.write(f"{i},{label}\n")
    print(f"Submission written: {path} ({len(labels_str)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pipeline.gbdt_model import OUTPUT_DIR, load_model

    output_dir = OUTPUT_DIR
    load_model(output_dir)  # Verify automatic-label training provenance before reuse.
    emissions_path = os.path.join(output_dir, "emissions.npy")

    if not os.path.exists(emissions_path):
        print("ERROR: emissions.npy not found. Run gbdt_model.py first.")
        sys.exit(1)

    emissions = np.load(emissions_path)
    print(f"Loaded emissions: {emissions.shape}")

    # Build transition matrix
    log_trans = build_transition_matrix()

    # Decode
    print("\nDecoding with Viterbi...")
    # No kickoff labels from ground truth; start each half with a uniform prior.
    labels = decode_match(emissions, log_trans)

    # Post-process
    labels = remove_short_segments(labels, min_frames=5)

    # Convert and write
    label_strings = labels_to_submission(labels)
    sub_path = os.path.join(ROOT, "evaluation", "results", "weak_predictions.csv")
    write_submission(label_strings, sub_path)

    print("\nDone! Score it with:")
    print("  python evaluation/run_evaluation.py")

