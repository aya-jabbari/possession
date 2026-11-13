"""Run all released checkpoints with identical, frozen ATD preprocessing.

Prediction only: python -m pathcrf_pipeline.benchmark
Then evaluate separately: python -m pathcrf_pipeline.compare_checkpoints
"""
import argparse
import json
import time

from . import OUTPUT
from .prepare import load_tracking, file_sha256
from .rules import build_rules
from .run import execute, install_ground_truth_guard


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',nargs='+',type=int,choices=[100,110,120,130,140],default=[100,110,120,130,140])
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if len(set(args.trials)) != len(args.trials):
        raise ValueError('Duplicate checkpoint IDs')
    install_ground_truth_guard()
    data=load_tracking()
    rules=build_rules(data)
    folder=OUTPUT/'checkpoints'
    folder.mkdir(parents=True,exist_ok=True)
    plan={'trials':args.trials,'threads':args.threads,'min_team_players':8,
          'cpu_flush_denormal':True,
          'goalkeeper_mode':'positional_per_window','new_tracker_used':False,
          'ground_truth_access_guard_enabled':True,
          'tracking_provenance':json.loads((OUTPUT/'tracking_meta.json').read_text()),
          'preprocessing_sha256':file_sha256(__import__('pathcrf_pipeline.prepare',fromlist=['']).__file__),
          'rules_sha256':file_sha256(__import__('pathcrf_pipeline.rules',fromlist=['']).__file__)}
    (folder/'benchmark_plan.json').write_text(json.dumps(plan,indent=2))
    fingerprint=None
    coverage=None
    rules_hash=None
    start=time.perf_counter()
    for trial in args.trials:
        print(f'\nStarting checkpoint {trial}',flush=True)
        destination=folder/str(trial)
        summary=execute(trial=trial,threads=args.threads,output_dir=destination,data=data,rules=rules)
        current_rules=file_sha256(destination/'results'/'rules_only.csv')
        if fingerprint is not None:
            if summary['input_fingerprint_sha256'] != fingerprint:
                raise RuntimeError('Checkpoint input tensors differ')
            if summary['model_covered_graded_frames'] != coverage or current_rules != rules_hash:
                raise RuntimeError('Checkpoint coverage/rule inputs differ')
        fingerprint=summary['input_fingerprint_sha256']
        coverage=summary['model_covered_graded_frames']
        rules_hash=current_rules
    plan.update({'completed_trials':args.trials,'seconds':time.perf_counter()-start,
                 'input_fingerprint_sha256':fingerprint,'rules_csv_sha256':rules_hash,
                 'model_covered_graded_frames':coverage})
    (folder/'benchmark_plan.json').write_text(json.dumps(plan,indent=2))
    print('All requested checkpoints completed with identical input fingerprints and coverage.',flush=True)


if __name__=='__main__':
    main()
