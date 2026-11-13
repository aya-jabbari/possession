"""Automatic training targets from tracking/Smart Tagging, never answer keys.

Unknown frames are excluded. A missing ball is not evidence of a dead ball.
Thresholds are physical heuristics, not selected by evaluation scores.
"""
import numpy as np
import pandas as pd
from games import GAME, GAMES

LABEL_SOURCE = "tracking_smart_tagging_v1"
RULES = {
    "control_distance_m": 2.0, "opponent_margin_m": 1.5,
    "max_control_speed_m_s": 25.0, "tag_distance_m": 5.0, "tag_margin_m": 0.5,
    "outside_margin_m": 0.5, "stationary_speed_m_s": 0.5,
    "stationary_seconds": 2.0, "unattended_distance_m": 5.0,
}


def sustained(mask, minimum):
    """Keep complete contiguous True runs of at least minimum frames."""
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    result = np.zeros(len(mask), dtype=bool)
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if end - start >= minimum:
            result[start:end] = True
    return result


def build_weak_labels(features):
    """Return label (-1 unknown, 0 H, 1 A, 2 D), weight and rule per frame."""
    required = ["half", "ball_visible", "ball_x", "ball_y", "ball_speed",
                "nearest_home_dist", "nearest_away_dist",
                "st_is_offensive_home", "st_is_offensive_away"]
    missing = set(required) - set(features.columns)
    if missing:
        raise ValueError(f"Missing weak-label features: {sorted(missing)}")
    if not features.index.equals(pd.RangeIndex(len(features))):
        raise ValueError("Weak labels require contiguous video frames starting at zero")
    n = len(features)
    labels = np.full(n, -1, dtype=np.int8)
    weights = np.zeros(n, dtype=np.float32)
    rules = np.full(n, "unknown", dtype="<U24")
    half = features["half"].to_numpy()
    active = np.isin(half, [1, 2])
    x, y = features["ball_x"].to_numpy(), features["ball_y"].to_numpy()
    speed = features["ball_speed"].to_numpy()
    visible = (features["ball_visible"].to_numpy() == 1) & np.isfinite(x) & np.isfinite(y)
    home = features["nearest_home_dist"].to_numpy()
    away = features["nearest_away_dist"].to_numpy()
    geometry = np.isfinite(home) & np.isfinite(away)
    inside = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
    live = active & visible & geometry & inside & (speed <= RULES["max_control_speed_m_s"])
    home_tag = features["st_is_offensive_home"].to_numpy() == 1
    away_tag = features["st_is_offensive_away"].to_numpy() == 1
    for label, own, other, tag, opposing_tag in [
        (0, home, away, home_tag, away_tag), (1, away, home, away_tag, home_tag),
    ]:
        tagged = (live & tag & ~opposing_tag & (own <= RULES["tag_distance_m"])
                  & (other - own >= RULES["tag_margin_m"]))
        labels[tagged], weights[tagged], rules[tagged] = label, 0.6, "tag_and_geometry"
        controlled = (live & ~opposing_tag & (own <= RULES["control_distance_m"])
                      & (other - own >= RULES["opponent_margin_m"]))
        labels[controlled], weights[controlled], rules[controlled] = label, 1.0, "close_control"
    length, width = GAMES[GAME]["pitch"]
    margin = RULES["outside_margin_m"]
    outside = active & visible & ((x < -margin / length) | (x > 1 + margin / length)
                                  | (y < -margin / width) | (y > 1 + margin / width))
    stationary = (active & visible & geometry & (speed <= RULES["stationary_speed_m_s"])
                  & (np.minimum(home, away) > RULES["unattended_distance_m"]))
    stopped = np.zeros(n, dtype=bool)
    minimum = int(RULES["stationary_seconds"] * GAMES[GAME]["fps"])
    for half_number in [1, 2]:
        stopped |= sustained(stationary & (half == half_number), minimum)
    labels[stopped], weights[stopped], rules[stopped] = 2, 0.6, "stationary_unattended"
    labels[outside], weights[outside], rules[outside] = 2, 1.0, "outside_pitch"
    return pd.DataFrame({"label": labels, "weight": weights, "rule": rules}, index=features.index)
