"""Reproduce model, rule-only and combined full-length predictions without GT.

Run from possession/: python -m pathcrf_pipeline.run
"""
import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import OUTPUT
from .prepare import load_tracking
from .rules import build_rules, combine_predictions


def install_ground_truth_guard():
    """Enforce prediction/evaluation separation, including indirect file opens."""
    def guard(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, Path)):
            path = str(args[0]).replace("\\", "/").lower()
            if "/evaluation/gt/" in path or path.startswith("evaluation/gt/"):
                raise PermissionError("Prediction is forbidden from opening ground truth: " + path)
    sys.addaudithook(guard)


def write_submission(path, labels):
    labels = np.asarray(labels)
    if not np.isin(labels,list("HADX")).all():
        raise ValueError("Invalid submission labels")
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"frame":np.arange(len(labels)),"label":labels}).to_csv(path,index=False)


def execute(trial=140, threads=4, min_team_players=8, max_windows=None,
            output_dir=None, data=None, rules=None):
    """Predict without evaluation. Callers install the GT guard before loading data."""
    t0=time.perf_counter()
    if data is None:
        print('Loading/alignment-checking tactical ATD...',flush=True)
        data=load_tracking()
    if rules is None:
        print('Building automatic control and stoppage evidence...',flush=True)
        rules=build_rules(data)
    from .infer import run_model
    votes,weights,summary=run_model(data,rules,min_team_players,max_windows,threads,trial,output_dir)
    if summary['accepted_windows'] == 0:
        raise RuntimeError('No usable model windows; refusing to present a rules-only run as PathCRF')
    pure,combined=combine_predictions(votes,weights,rules)
    destination=Path(output_dir) if output_dir is not None else OUTPUT
    if max_windows is not None:
        destination=destination/"smoke"
    results=destination/"results"
    for name,labels in [("pathcrf_model",pure),("rules_only",rules['labels']),('pathcrf_rules',combined)]:
        write_submission(results/f'{name}.csv',labels)
    frame_audit=pd.DataFrame({'video_frame':data['frame'],'atd_frame_id':data['frame']+1,
                              'in_half':rules['active'],'estimated_dead':rules['dead'],
                              'model_covered':weights>0,'model_label':pure,
                              'rule_label':rules['labels'],'combined_label':combined})
    frame_audit.to_parquet(destination/'frame_audit.parquet',index=False)
    summary['total_pipeline_seconds']=time.perf_counter()-t0
    summary['result_directory']=str(results)
    summary['label_counts']={name:{v:int(np.sum(labels==v)) for v in 'HADX'}
                             for name,labels in [('model',pure),('rules',rules['labels']),('combined',combined)]}
    summary['ground_truth_access_guard_enabled']=True
    summary['tracking_provenance']=json.loads((OUTPUT/'tracking_meta.json').read_text())
    (destination/'run_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--min-team-players",type=int,default=8)
    parser.add_argument("--max-windows",type=int)
    parser.add_argument("--trial",type=int,choices=[100,110,120,130,140],default=140)
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args()
    install_ground_truth_guard()
    destination=args.output_dir
    if destination is None and args.trial != 140:
        destination=OUTPUT/'checkpoints'/str(args.trial)
    execute(args.trial,args.threads,args.min_team_players,args.max_windows,destination)


if __name__=='__main__':
    main()
