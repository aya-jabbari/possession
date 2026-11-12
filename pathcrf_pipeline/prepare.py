"""Read tactical ATD and screen model windows for continuity and finite inputs."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.signal import savgol_filter

from games import GAME, GAMES, asset_paths, half_video_blocks
from . import OUTPUT

FPS = 25
PITCH = np.array([105.0, 68.0], dtype=np.float32)
FEATURE_ORDER = ["team_indicator", "goalkeeper_indicator", "x", "y",
                 "vx", "vy", "speed", "accel"]
PREP_VERSION = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runs(mask):
    """Half-open intervals of a one-dimensional boolean mask."""
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def load_tracking(rebuild=False):
    """Strict 1-based ATD to 0-based video alignment, cached without answer keys."""
    paths = asset_paths()
    raw, meta = Path(paths["atd_raw"]), Path(paths["atd_meta"])
    provenance = {"version": PREP_VERSION, "raw_sha256": file_sha256(raw),
                  "metadata_sha256": file_sha256(meta)}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = OUTPUT / "tracking.npz"
    sidecar = OUTPUT / "tracking_meta.json"
    if not rebuild and cache.exists() and sidecar.exists():
        if json.loads(sidecar.read_text()) == provenance:
            with np.load(cache) as data:
                return {name: data[name] for name in data.files}
    xml = ET.parse(meta).getroot()
    team_ids = [t.get("id") for t in xml.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("Expected two teams in ATD metadata")
    players = list(xml.iter("Player"))
    team = np.array([team_ids.index(p.get("teamId")) for p in players], np.int8)
    n, k = GAMES[GAME]["n_video_frames"], len(players)
    positions = np.full((n, k, 2), np.nan, np.float32)
    speed = np.full((n, k), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    seen = np.zeros(n, bool)
    with raw.open() as handle:
        for line in handle:
            parts = line.rstrip().split(":")
            if len(parts) != 3:
                raise ValueError("Malformed ATD row")
            frame = int(parts[0]) - 1
            if frame < 0 or frame >= n or seen[frame]:
                raise ValueError("Out-of-range or duplicate ATD frame")
            values = np.fromstring(parts[1].rstrip(";").replace(";", ","),
                                   sep=",", dtype=np.float32)
            if len(values) != k * 3:
                raise ValueError(f"ATD frame {frame}: expected {k} x,y,speed groups")
            values = values.reshape(k, 3)
            positions[frame], speed[frame] = values[:, :2], values[:, 2]
            b = np.fromstring(parts[2], sep=",", dtype=np.float32)
            if len(b) < 2:
                raise ValueError("Missing ball fields")
            ball[frame] = b[:2]
            seen[frame] = True
    if not seen.all():
        raise ValueError("ATD does not cover every tactical video frame")
    result = {"positions": positions, "raw_speed": speed, "ball": ball,
              "team": team, "frame": np.arange(n, dtype=np.int32)}
    np.savez_compressed(cache, **result)
    sidecar.write_text(json.dumps(provenance, indent=2))
    return result


def fill_short_gaps(xy, max_gap=5):
    """Interpolate only short internal gaps with observed endpoints."""
    out = np.array(xy, dtype=np.float32, copy=True)
    good = np.isfinite(out).all(axis=1)
    for start, end in runs(~good):
        if start > 0 and end < len(out) and end - start <= max_gap:
            out[start:end] = np.linspace(out[start-1], out[end], end-start+2)[1:-1]
    return out


def movement_features(xy_metres, fps=FPS):
    """Match upstream derivative smoothing; acceleration is vector magnitude."""
    xy = np.asarray(xy_metres, dtype=np.float32)
    if len(xy) < 3 or not np.isfinite(xy).all():
        raise ValueError("Movement features require finite contiguous positions")
    velocity = np.diff(xy, axis=0) * fps
    if len(velocity) >= 15:
        velocity = savgol_filter(velocity, 15, 2, axis=0)
    acceleration = np.diff(velocity, axis=0) * fps
    if len(acceleration) >= 9:
        acceleration = savgol_filter(acceleration, 9, 2, axis=0)
    result = np.zeros((len(xy), 6), np.float32)
    result[:, :2] = xy
    result[1:, 2:4] = velocity
    result[0, 2:4] = velocity[0]
    result[:, 4] = np.linalg.norm(result[:, 2:4], axis=1)
    result[1:-1, 5] = np.linalg.norm(acceleration, axis=1)
    return result


def window_schedule(blocks=None, length=250, stride=125):
    """Overlap within halves, include the final frame, never bridge halftime."""
    blocks = half_video_blocks(GAME) if blocks is None else blocks
    windows = []
    for half, first, last in blocks:
        half, first, last = int(half), int(first), int(last)
        if last-first+1 <= length:
            windows.append((half, first, last+1))
            continue
        starts = list(range(first, last-length+2, stride))
        if starts[-1] != last-length+1:
            starts.append(last-length+1)
        windows.extend((half, start, start+length) for start in starts)
    return windows


def make_window(positions, team, min_team_players=8, max_gap=5):
    """Use stable observed slots, verified duplicate removal and explicit padding.

    No global player re-identification is claimed. A slot with a >2m step or a
    long gap is excluded from this window. No arbitrary first-22 truncation.
    """
    candidates = []
    for slot in range(positions.shape[1]):
        xy = fill_short_gaps(positions[:, slot], max_gap)
        if not np.isfinite(xy).all():
            continue
        metres = xy * PITCH
        if np.any(np.linalg.norm(np.diff(metres, axis=0), axis=1) > 2.0):
            continue
        features = movement_features(metres)
        if np.max(features[:, 4]) > 20.0:
            continue
        candidates.append((slot, features))
    teams, removed_duplicates = [], []
    for side in [0, 1]:
        kept = []
        for slot, feats in candidates:
            if team[slot] != side:
                continue
            duplicate = next((other for other, previous in kept
                              if np.mean(np.linalg.norm(feats[:, :2]-previous[:, :2], axis=1) < 0.75) > 0.8), None)
            if duplicate is None:
                kept.append((slot, feats))
            else:
                removed_duplicates.append([int(slot), int(duplicate)])
        if len(kept) < min_team_players:
            return None, {"reason": "insufficient_continuous_players", "team": side,
                          "count": len(kept)}
        if len(kept) > 11:
            return None, {"reason": "unresolved_excess_players", "team": side,
                          "count": len(kept)}
        teams.append(kept)
    # Same positional keeper heuristic as upstream, applied per continuous window.
    keepers = [max(items, key=lambda item: abs(item[1][:, 0].mean()-52.5))[0] for items in teams]
    keeper_x = [next(f[:, 0].mean() for s, f in items if s == keeper)
                for items, keeper in zip(teams, keepers)]
    left = int(keeper_x[1] < keeper_x[0])
    order = [left, 1-left]
    tensor = np.zeros((len(positions), 26, 8), np.float32)
    node_team = np.full(26, -1, np.int8)
    node_slot = np.full(26, -1, np.int16)
    active = np.zeros(26, bool)
    for block, side in enumerate(order):
        for j, (slot, feats) in enumerate(teams[side]):
            node = block*11+j
            tensor[:, node, 0] = -1.0 if block == 0 else 1.0
            tensor[:, node, 1] = float(slot == keepers[side])
            tensor[:, node, 2:] = feats
            node_team[node], node_slot[node], active[node] = side, slot, True
    outside = np.array([[0, 34], [105, 34], [52.5, 0], [52.5, 68]], np.float32)
    tensor[:, 22:, 2:4] = outside
    active[22:] = True
    return {"tensor": tensor, "node_team": node_team, "node_slot": node_slot,
            "active": active}, {"reason": "accepted", "players_home": len(teams[0]),
                                "players_away": len(teams[1]), "keepers": keepers,
                                "duplicates_removed": removed_duplicates,
                                "padded_nodes": 22-sum(map(len, teams))}

