"""Pretrained forward pass and official dynamic masked CRF decoding."""
import json
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch

from . import ASSETS, OUTPUT
from .prepare import FEATURE_ORDER, file_sha256, make_window, window_schedule
from .rules import segments_from_rules


def load_model(threads=4):
    from models.utils import build_model
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    folder = ASSETS / "saved" / "140"
    args = json.loads((folder/"args.json").read_text())
    if args["node_in_dim"] != 8 or args["team_size"] != 11:
        raise ValueError("Unexpected checkpoint schema")
    model = build_model(args, device=torch.device("cpu"))
    weights = folder/"model"/"state_dict_best_acc.pt"
    checkpoint_hash = file_sha256(weights)
    if checkpoint_hash != "55757d9dca9777d60356ba368dc315867ea1ae38e90c332e935b9c5be72d83c9":
        raise ValueError("Checkpoint differs from the verified official saved/140 weights")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, {"checkpoint_sha256":checkpoint_hash,
                   "source_commit":"1b7c6f85a6b78c92e4543fde72ca2b86a933f997",
                   "args":args,"torch_version":torch.__version__,"device":"cpu",
                   "threads":threads}


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
    valid = model.valid_orig_edge_ids
    emissions = logits.index_select(2, valid)
    embeds = embeddings.index_select(2, valid)
    src = model.comp_src.cpu().numpy()
    dst = model.comp_dst.cpu().numpy()
    # Upstream pads absent players with zero vectors. Prevent the decoder from
    # assigning the ball to a padding node, which is not a physical player.
    allowed_nodes = prepared["active"]
    valid_states = allowed_nodes[src] & allowed_nodes[dst]
    emissions[:, :, ~torch.from_numpy(valid_states)] = -1e4
    if not torch.isfinite(emissions).all() or not torch.isfinite(embeds).all():
        raise RuntimeError("Non-finite checkpoint output")
    decoded = model.crf.decode(emissions, embeds).squeeze(0).cpu().numpy()
    if not valid_states[decoded].all():
        raise RuntimeError("Decoder selected a padding player")
    # Validate hard CRF constraints, including absorbing boundary states.
    inc = model.crf.inc_idx.cpu().numpy()
    mask = model.crf.inc_mask.cpu().numpy()
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


def run_model(data, rules, min_team_players=8, max_windows=None, threads=4):
    if not 1 <= min_team_players <= 11:
        raise ValueError("min_team_players must be between 1 and 11")
    destination = OUTPUT / "smoke" if max_windows is not None else OUTPUT
    model, model_info = load_model(threads)
    schedule = window_schedule(segments_from_rules(rules))
    if max_windows is not None:
        schedule = schedule[:max_windows]
    n = len(data["frame"])
    votes, weights = np.zeros((n,3),np.float32), np.zeros(n,np.float32)
    audit, edges = [], []
    t0 = time.perf_counter()
    for i,(half,start,end) in enumerate(schedule):
        prepared, info = make_window(data["positions"][start:end],data["team"],min_team_players)
        row = {"half":half,"start":start,"end_exclusive":end,**info}
        if prepared is not None:
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
            print(f"PathCRF {i+1}/{len(schedule)} windows, {accepted} accepted, {time.perf_counter()-t0:.1f}s",flush=True)
    destination.mkdir(parents=True,exist_ok=True)
    # Save the expensive inference arrays before human-readable exports.
    np.savez_compressed(destination/"model_votes.npz",votes=votes,weights=weights)
    (destination/"window_audit.json").write_text(json.dumps(audit,indent=2))
    pd.DataFrame(edges).to_parquet(destination/"decoded_edges.parquet",index=False)
    accepted = [r for r in audit if r["reason"] == "accepted"]
    active = rules["active"]
    summary={**model_info,"feature_order":FEATURE_ORDER,"min_players_per_team":min_team_players,
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

