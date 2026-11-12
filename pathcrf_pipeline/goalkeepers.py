"""Independent per-frame YOLO detection and same-frame ATD association.

No model.track(), persistent IDs, new tracker or cross-frame association is used.
"""
import argparse
import json
import time

import numpy as np

from games import asset_paths
from starter.load_homography import load, unproject
from . import ASSETS, OUTPUT
from .prepare import load_tracking, PITCH


def match_detection(box, matrix, positions, max_distance=4.0, min_margin=1.0):
    """Bottom-centre ground approximation; return no match for ambiguity."""
    x1, y1, x2, y2 = map(float, box)
    xy = np.array(unproject(matrix, (x1+x2)/2, y2), dtype=float)
    if not np.isfinite(xy).all():
        return -1, float("nan"), xy
    distance = np.linalg.norm((positions-xy)*PITCH, axis=1)
    valid = np.flatnonzero(np.isfinite(distance))
    if not len(valid):
        return -1, float("nan"), xy
    ordered = valid[np.argsort(distance[valid])]
    best = ordered[0]
    margin = np.inf if len(ordered) == 1 else distance[ordered[1]]-distance[best]
    slot = int(best) if distance[best] <= max_distance and margin >= min_margin else -1
    return slot, float(distance[best]), xy


def detect_clips(intervals, imgsz=640, batch_size=4, threads=4):
    import cv2
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(threads)
    data = load_tracking()
    matrices, errors, valid = load()
    weights = ASSETS / "detector" / "yolo-football-player-detection.pt"
    model = YOLO(str(weights))
    goalkeeper_ids = [i for i, name in model.names.items() if name.lower() == "goalkeeper"]
    if len(goalkeeper_ids) != 1:
        raise ValueError(f"Expected one goalkeeper class, found {model.names}")
    paths = asset_paths()
    cap = cv2.VideoCapture(str(paths["video"]))
    if not cap.isOpened():
        raise RuntimeError("Cannot open tactical video")
    if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(data["frame"]):
        raise ValueError("Video and ATD lengths disagree")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / "goalkeeper_detections.jsonl"
    diagnostics = []
    elapsed_start = time.perf_counter()
    with destination.open("w", encoding="utf-8") as handle:
        for start, end in intervals:
            if not (0 <= start < end <= len(data["frame"])):
                raise ValueError("Invalid clip frame interval")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            if int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != start:
                raise RuntimeError(f"Video seek did not reach requested frame {start}")
            current = start
            while current < end:
                images, frames = [], []
                for _ in range(min(batch_size, end-current)):
                    ok, image = cap.read()
                    if not ok:
                        raise RuntimeError(f"Video decode failed at frame {current}")
                    images.append(image)
                    frames.append(current)
                    current += 1
                results = model.predict(images, imgsz=imgsz, classes=goalkeeper_ids,
                                        conf=0.25, device="cpu", verbose=False)
                for frame, image, result in zip(frames, images, results):
                    found, matched = [], []
                    for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                        slot, distance, xy = match_detection(box, matrices[frame], data["positions"][frame]) if valid[frame] else (-1, float("nan"), np.full(2, np.nan))
                        found.append({"xyxy":box.tolist(), "confidence":float(score),
                                      "pitch_xy":[float(v) if np.isfinite(v) else None for v in xy], "slot":slot,
                                      "distance_m":distance if np.isfinite(distance) else None,
                                      "team":int(data["team"][slot]) if slot >= 0 else -1})
                        if slot >= 0:
                            matched.append(slot)
                    row = {"video_frame":frame,"atd_frame_id":frame+1,
                           "homography_valid":bool(valid[frame]),
                           "reprojection_error":float(errors[frame]) if np.isfinite(errors[frame]) else None,"detections":found}
                    handle.write(json.dumps(row, allow_nan=False)+"\n")
                    diagnostics.append((frame,len(found),len(matched)))
                    if frame == start:
                        for item in found:
                            x1,y1,x2,y2=map(int,item["xyxy"])
                            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
                            cv2.circle(image,((x1+x2)//2,y2),5,(0,0,255),-1)
                            cv2.putText(image,f"GK {item['confidence']:.2f} slot={item['slot']}",(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2)
                        cv2.imwrite(str(OUTPUT/f"goalkeeper_frame_{frame}.jpg"),image)
                print(f"YOLO frames {frames[0]}..{frames[-1]} ({len(diagnostics)} processed)",flush=True)
    cap.release()
    summary={"intervals_half_open":intervals,"frames_processed":len(diagnostics),
             "frames_with_detection":sum(n>0 for _,n,_ in diagnostics),
             "frames_with_atd_match":sum(n>0 for _,_,n in diagnostics),
             "detections":sum(n for _,n,_ in diagnostics),
             "matched_detections":sum(n for _,_,n in diagnostics),
             "seconds":time.perf_counter()-elapsed_start,"imgsz":imgsz,
             "tracker_used":False,"full_match_yolo":False,
             "matching":"same-frame bottom-centre/inverse-homography; <=4m and >=1m nearest margin"}
    (OUTPUT/"goalkeeper_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips",nargs="+",default=["12000:12025","45000:45025","113742:113767","150000:150025"])
    parser.add_argument("--imgsz",type=int,default=640)
    args=parser.parse_args()
    detect_clips([tuple(map(int,item.split(":"))) for item in args.clips],args.imgsz)
