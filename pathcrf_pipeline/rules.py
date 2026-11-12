"""Fixed, automatic possession/stoppage heuristics; no answer-key access."""
import numpy as np

from games import GAME, half_video_blocks
from starter.load_smart_tagging import load as load_tags
from .prepare import FPS, PITCH, runs

RULE_VERSION = 1


def retain_runs(mask, minimum):
    out = np.zeros(len(mask), bool)
    for start, end in runs(mask):
        if end-start >= minimum:
            out[start:end] = True
    return out


def build_rules(data):
    """Same basic control distances as the existing weak-supervision baseline.

    Missing ball never means dead. Confirmed outside observations may bridge at
    most 30 seconds to the next inside/moving, player-adjacent ball observation.
    These are imperfect estimated episodes, not referee/ground-truth states.
    """
    xy = data["positions"] * PITCH
    ball = data["ball"] * PITCH
    n = len(ball)
    visible = np.isfinite(ball).all(axis=1)
    distance = np.linalg.norm(xy-ball[:, None, :], axis=2)
    distance[~np.isfinite(distance)] = np.inf
    nearest = np.stack([np.min(distance[:, data["team"] == side], axis=1)
                        for side in [0, 1]], axis=1)
    speed = np.r_[np.nan, np.linalg.norm(np.diff(ball, axis=0), axis=1)*FPS]
    inside = visible & (ball[:, 0] >= 0) & (ball[:, 0] <= 105) & (ball[:, 1] >= 0) & (ball[:, 1] <= 68)
    outside = visible & ((ball[:, 0] < -0.5) | (ball[:, 0] > 105.5)
                         | (ball[:, 1] < -0.5) | (ball[:, 1] > 68.5))
    offensive = np.zeros((n, 2), bool)
    set_piece = np.zeros(n, bool)
    tags = load_tags()
    codes = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
             "BALL IN FINAL THIRD", "BALL IN THE BOX", "LONG BALL"}
    for event in tags:
        a, b = max(0, event["start_frame"]), min(n, event["end_frame"])
        code = str(event["code"]).upper()
        team = str(event["tags"].get("Team", "")).lower()
        side = 0 if "barcelona" in team else (1 if "atletic" in team or "athletic" in team else -1)
        if code in codes and side >= 0:
            offensive[a:b, side] = True
        if code == "SET PIECES":
            set_piece[a:b] = True
    control = np.full(n, -1, np.int8)
    for side in [0, 1]:
        with np.errstate(invalid="ignore"):
            margin = nearest[:, 1-side]-nearest[:, side]
        close = (nearest[:, side] <= 2) & (margin >= 1.5)
        tagged = offensive[:, side] & ~offensive[:, 1-side] & (nearest[:, side] <= 5) & (margin >= .5)
        evidence = inside & (speed <= 25) & (close | tagged)
        # A fleeting touch is not automatically controlled possession.
        control[retain_runs(evidence, 5)] = side
    active = np.zeros(n, bool)
    dead = np.zeros(n, bool)
    for half, first, last in half_video_blocks(GAME):
        active[first:last+1] = True
        stopped = (visible[first:last+1] & (speed[first:last+1] <= .5)
                   & (nearest[first:last+1].min(axis=1) > 5))
        dead[first:last+1] |= retain_runs(stopped, 2*FPS)
        dead[first:last+1] |= retain_runs(outside[first:last+1], 2)
        # Stop carrying an exit at the next supported restart, not at missing data.
        exits = runs(dead[first:last+1])
        for a, b in exits:
            a, b = a+first, b+first
            if not outside[a:b].any():
                continue
            limit = min(last+1, b+30*FPS)
            restart = np.flatnonzero(inside[b:limit] & (speed[b:limit] > 1)
                                    & (nearest[b:limit].min(axis=1) <= 4))
            end = b+int(restart[0]) if len(restart) else b
            dead[a:end] = True
    dead &= active
    rules = np.full(n, "X", dtype="<U1")
    for _, first, last in half_video_blocks(GAME):
        times = np.arange(first, last+1)
        observed = np.flatnonzero(control[first:last+1] >= 0)
        if len(observed):
            previous = np.maximum(0, np.searchsorted(observed, times-first, side="right")-1)
            rules[times] = np.array(["H", "A"])[control[first+observed[previous]]]
        rules[times[dead[times]]] = "D"
    return {"labels":rules,"dead":dead,"active":active,"nearest":nearest,
            "ball_speed":speed,"control":control,"set_piece":set_piece,
            "rule_version":RULE_VERSION}


def segments_from_rules(rules):
    blocks = []
    for half, first, last in half_video_blocks(GAME):
        for a, b in runs(~rules["dead"][first:last+1]):
            if b-a >= 25:
                blocks.append((half, first+a, first+b-1))
    return blocks


def combine_predictions(votes, weights, rules):
    """Three explicit outputs: partial model, rule-only and model+rules/fallback."""
    covered = weights > 0
    pure = np.full(len(weights), "X", dtype="<U1")
    pure[covered] = np.array(["H", "A", "D"])[votes[covered].argmax(axis=1)]
    hybrid = rules["labels"].copy()
    hybrid[covered] = pure[covered]
    hybrid[rules["dead"]] = "D"
    hybrid[~rules["active"]] = "X"
    return pure, hybrid

