"""Pretrained forward passes with each checkpoint's native decoder."""
import hashlib
import json
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch

from . import ASSETS, OUTPUT
from .prepare import FEATURE_ORDER, file_sha256, make_window, window_schedule
from .rules import segments_from_rules

TRIAL_NAMES = {100: "Non-CRF", 110: "Static Dense CRF", 120: "Static Masked CRF",
               130: "Dynamic Dense CRF", 140: "Dynamic Masked CRF"}
CHECKPOINT_SHA256 = {
    100: "8b9ddfa6c8e7510ca91320c12a187cd1d8e21822984216534b9a1ffd8c09428b",
    110: "112cdc05a2eb926d383d91c3f93c0540f63bebdae80a34e064485b6c89c74338",
    120: "d4ad02c1923cd0bf2d2f6566a6754001941293358cf6adffc127f3be26f07b11",
    130: "82eb3a3b96b82b1ae91145a295a3a34b85a6e9b1d330044bd968a5e9c05d88fc",
    140: "55757d9dca9777d60356ba368dc315867ea1ae38e90c332e935b9c5be72d83c9",
}


def load_model(threads=4, trial=140):
    from models.utils import build_model
    if trial not in TRIAL_NAMES:
        raise ValueError(f"Unsupported checkpoint: {trial}")
    torch.set_num_threads(threads)
    # Several released checkpoints contain subnormal (near-zero) weights. On
    # this CPU they trigger a large slow path. Apply one fixed execution policy
    # to every variant; parity tests check logits and decoded paths explicitly.
    flush_supported = torch.set_flush_denormal(True)
    torch.manual_seed(0)
    folder = ASSETS / "saved" / str(trial)
    args = json.loads((folder/"args.json").read_text())
    if args["trial"] != trial or args["node_in_dim"] != 8 or args["team_size"] != 11:
        raise ValueError("Unexpected checkpoint schema")
    target_fps = args.get("target_fps", args.get("fps", 25)/args.get("sample_freq", 5))
    if target_fps != 5 or args["window_seconds"] != 10:
        raise ValueError("Checkpoint sampling/window differs from the fixed adapter")
    model = build_model(args, device=torch.device("cpu"))
    weights = folder/"model"/"state_dict_best_acc.pt"
    checkpoint_hash = file_sha256(weights)
    if checkpoint_hash != CHECKPOINT_SHA256[trial]:
        raise ValueError(f"Checkpoint differs from the verified official saved/{trial} weights")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, {"trial":trial,"variant":TRIAL_NAMES[trial],"checkpoint_sha256":checkpoint_hash,
                   "source_commit":"1b7c6f85a6b78c92e4543fde72ca2b86a933f997",
                   "args":args,"torch_version":torch.__version__,"device":"cpu",
                   "threads":threads,"cpu_flush_denormal":bool(flush_supported)}


@torch.inference_mode()
def predict_window(model, prepared):
    tensor = prepared["tensor"]
    # 25 -> 5 FPS exactly once; include the last observation as upstream does.
    selected = np.arange(0, len(tensor), 5)
    if selected[-1] != len(tensor)-1:
        selected = np.r_[selected, len(tensor)-1]
    x = torch.from_numpy(tensor[selected]).unsqueeze(0)
    if not torch.isfinite(x).all() or x.shape[2:] != (26, 8):
        raise ValueError("Invalid model input")
    macro, logits, embeddings = model(x)
    crf = model.crf
    if crf is None:
        # The published Non-CRF baseline uses independent argmax over all 26²
        # states. Do not add CRF/path constraints to this baseline.
        emissions = logits.clone()
        src = np.repeat(np.arange(26), 26)
        dst = np.tile(np.arange(26), 26)
        embeds = None
    else:
        valid = model.valid_orig_edge_ids
        emissions = logits.index_select(2, valid)
        embeds = embeddings.index_select(2, valid) if embeddings is not None else None
        src = model.comp_src.cpu().numpy()
        dst = model.comp_dst.cpu().numpy()
    # Upstream pads absent players with zero vectors. Prevent the decoder from
    # assigning the ball to a padding node, which is not a physical player.
    allowed_nodes = prepared["active"]
    valid_states = allowed_nodes[src] & allowed_nodes[dst]
    emissions[:, :, ~torch.from_numpy(valid_states)] = -1e4
    if not torch.isfinite(emissions).all() or (embeds is not None and not torch.isfinite(embeds).all()):
        raise RuntimeError("Non-finite checkpoint output")
    if crf is None:
        decoded = emissions.argmax(dim=-1).squeeze(0).cpu().numpy()
    elif hasattr(crf, "edge_embed_dim"):
        if embeds is None:
            raise RuntimeError("Dynamic CRF is missing edge embeddings")
        decoded = crf.decode(emissions, embeds).squeeze(0).cpu().numpy()
    else:
        decoded = crf.decode(emissions).squeeze(0).cpu().numpy()
    if not valid_states[decoded].all():
        raise RuntimeError("Decoder selected a padding player")
    # Only masked variants impose these hard constraints. Dense and Non-CRF
    # variants intentionally allow transitions that masked models disallow.
    if crf is not None and hasattr(crf, "inc_idx"):
        inc = crf.inc_idx.cpu().numpy()
        mask = crf.inc_mask.cpu().numpy()
        if any(not np.any((inc[b] == a) & mask[b]) for a,b in zip(decoded[:-1],decoded[1:])):
            raise RuntimeError("Decoded path violates a CRF transition constraint")
    source, destination = src[decoded], dst[decoded]
    # Sender retains team possession during flight. Opposing control changes
    # ownership when that player becomes the source of the subsequent state.
    classes = prepared["node_team"][source].astype(np.int8)
    classes[source >= 22] = 2
    if np.any(classes < 0):
        raise RuntimeError("Decoded unknown team")
    # Forward-hold categorical states between 5 FPS observations. The neural
    # attention and Viterbi pass themselves use future context within the window.
    index = np.searchsorted(selected, np.arange(len(tensor)), side="right")-1
    full_classes = classes[index]
    return full_classes, {"sample_frame":selected,"source_node":source,
                          "destination_node":destination,
                          "source_slot":prepared["node_slot"][source],
                          "destination_slot":prepared["node_slot"][destination],
                          "class":classes}


def run_model(data, rules, min_team_players=8, max_windows=None, threads=4,
              trial=140, output_dir=None):
    if not 1 <= min_team_players <= 11:
        raise ValueError("min_team_players must be between 1 and 11")
    from pathlib import Path
    destination = Path(output_dir) if output_dir is not None else OUTPUT
    if max_windows is not None:
        destination = destination / "smoke"
    model, model_info = load_model(threads, trial)
    schedule = window_schedule(segments_from_rules(rules))
    if max_windows is not None:
        schedule = schedule[:max_windows]
    n = len(data["frame"])
    votes, weights = np.zeros((n,3),np.float32), np.zeros(n,np.float32)
    audit, edges = [], []
    input_digest = hashlib.sha256()
    t0 = time.perf_counter()
    for i,(half,start,end) in enumerate(schedule):
        prepared, info = make_window(data["positions"][start:end],data["team"],min_team_players)
        row = {"half":half,"start":start,"end_exclusive":end,**info}
        input_digest.update(json.dumps(row,sort_keys=True).encode())
        if prepared is not None:
            for key in ["tensor", "node_team", "node_slot", "active"]:
                input_digest.update(prepared[key].tobytes())
            labels, path = predict_window(model,prepared)
            # Overlap reduces dependence on arbitrary 10s window boundaries.
            weight = np.minimum(np.arange(end-start)+1,np.arange(end-start,0,-1)).astype(np.float32)
            weight /= weight.max()
            votes[np.arange(start,end),labels] += weight
            weights[start:end] += weight
            for j,frame in enumerate(path["sample_frame"]):
                edges.append({"window":i,"video_frame":int(start+frame),
                              "source_node":int(path["source_node"][j]),
                              "destination_node":int(path["destination_node"][j]),
                              "source_slot":int(path["source_slot"][j]),
                              "destination_slot":int(path["destination_slot"][j]),
                              "label":"HAD"[path["class"][j]]})
            row["model_input_frames"] = len(path["sample_frame"])
        audit.append(row)
        if i%20 == 0 or i == len(schedule)-1:
            accepted = sum(r["reason"] == "accepted" for r in audit)
            print(f"PathCRF {trial}: {i+1}/{len(schedule)} windows, {accepted} accepted, {time.perf_counter()-t0:.1f}s",flush=True)
    destination.mkdir(parents=True,exist_ok=True)
    # Save the expensive inference arrays before human-readable exports.
    np.savez_compressed(destination/"model_votes.npz",votes=votes,weights=weights)
    (destination/"window_audit.json").write_text(json.dumps(audit,indent=2))
    pd.DataFrame(edges).to_parquet(destination/"decoded_edges.parquet",index=False)
    accepted = [r for r in audit if r["reason"] == "accepted"]
    active = rules["active"]
    summary={**model_info,"feature_order":FEATURE_ORDER,"min_players_per_team":min_team_players,
             "input_fingerprint_sha256":input_digest.hexdigest(),
             "decoder":"independent_argmax" if model.crf is None else type(model.crf).__name__,
             "maximum_internal_gap_frames":5,"window_count":len(audit),"accepted_windows":len(accepted),
             "window_reasons":dict(Counter(r["reason"] for r in audit)),
             "accepted_with_22_observed_players":sum(r["padded_nodes"]==0 for r in accepted),
             "accepted_with_padding":sum(r["padded_nodes"]>0 for r in accepted),
             "graded_frames":int(active.sum()),"model_covered_graded_frames":int(((weights>0)&active).sum()),
             "model_coverage_pct":float(100*((weights>0)&active).sum()/active.sum()),
             "estimated_dead_frames":int(rules["dead"].sum()),
             "seconds":time.perf_counter()-t0,"goalkeeper_mode":"positional_per_window",
             "yolo_used_for_full_match":False,"ground_truth_used_for_prediction":False,
             "limitations":["Automatic live/dead segmentation is imperfect",
                            "Windows reset the CRF path; overlap only blends team labels",
                            "No global repair of fragmented physical identities",
                            "Padding of absent players is a domain shift from high-quality training tracking",
                            "Goalkeeper identity uses the authorized positional fallback"]}
    (destination/"inference_summary.json").write_text(json.dumps(summary,indent=2))
    return votes,weights,summary

