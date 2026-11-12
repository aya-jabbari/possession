"""Evaluate frozen predictions and supplied baselines on PUBLIC ground truth only.

This module is intentionally separate from prediction and is never imported by it.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import ROOT, OUTPUT
from .prepare import file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap',type=int,default=1000)
    args=parser.parse_args()
    sys.path.insert(0,str(ROOT/'evaluation'))
    import run_evaluation as evaluator
    import possession_metrics as pm
    gt=ROOT/'evaluation'/'gt'/'BAR-ATH_possession_gt_public.csv'
    truth=pm.label_codes(pm.read_label_csv(str(gt)))
    candidates=sorted((ROOT/'evaluation'/'results').glob('baseline_*.csv'))
    candidates += [OUTPUT/'results'/f'{name}.csv'
                   for name in ['rules_only','pathcrf_model','pathcrf_rules']]
    destination=OUTPUT/'comparison'
    destination.mkdir(parents=True,exist_ok=True)
    rows=[]
    results={}
    for path in candidates:
        result,reason=evaluator.score_submission(str(path),truth,pm.TAUS,args.bootstrap)
        if reason:
            raise ValueError(f'{path.name}: {reason}')
        print(f'{path.stem}: macroF1={result["macro_f1"]:.4f}, spellF1={result["spell_f1_mean"]:.4f}',flush=True)
        row=evaluator.scoreboard_row(None,{'name':path.stem,'path':str(path),'result':result,'reason':None},pm.TAUS)
        row['prediction_sha256']=file_sha256(path)
        rows.append(row)
        report=evaluator.json_safe(result)
        (destination/f'{path.stem}.json').write_text(json.dumps(report,indent=2))
        results[path.stem]=pm.label_codes(pm.read_label_csv(str(path)))
    table=pd.DataFrame(rows).sort_values(['spell_f1_mean','macro_f1'],ascending=False)
    table['rank']=np.arange(1,len(table)+1)
    table.to_csv(destination/'scoreboard.csv',index=False)
    summary=json.loads((OUTPUT/'run_summary.json').read_text())
    summary['public_model_coverage_pct']=float(100*np.mean(pd.read_parquet(OUTPUT/'frame_audit.parquet')['model_covered'].to_numpy()[truth != pm.X]))
    summary['public_ground_truth_sha256']=file_sha256(gt)
    (destination/'comparison_meta.json').write_text(json.dumps(summary,indent=2))
    columns=['submission','macro_f1','spell_f1_mean','poss','n_abstain_on_graded']
    print(table[columns].to_string(index=False),flush=True)
    lines=['# PathCRF comparison: public five-minute window','',
           'All methods were evaluated on the same published 7,500 frames (17 possessions).',
           'Predictions were frozen before evaluation. No parameters were selected against these scores.',
           'These scores do not measure full-match accuracy; only the organisers hold the private labels.','',
           '| Method | macroF1 | spellF1 | Predicted possessions | Graded abstentions |',
           '| --- | ---: | ---: | ---: | ---: |']
    for _,r in table.iterrows():
        lines.append(f'| {r["submission"]} | {r["macro_f1"]:.4f} | {r["spell_f1_mean"]:.4f} | {int(r["poss"])} | {int(r["n_abstain_on_graded"])} |')
    lines += ['', '## What each PathCRF entry means','',
              '- `pathcrf_model`: checkpoint+CRF predictions on usable windows; X elsewhere. Automatic estimated-live segmentation is shared with the rules.',
              '- `rules_only`: independent control/stoppage heuristics; no neural predictions.',
              '- `pathcrf_rules`: checkpoint predictions where available, rule fallback on uncovered frames, and the same heuristic dead-ball override.',
              '',f'Full-match model coverage: {summary["model_coverage_pct"]:.2f}% of graded frames.',
              f'Public-window model coverage: {summary["public_model_coverage_pct"]:.2f}%.',
              f'Accepted windows with zero-padded absent players: {summary["accepted_with_padding"]}; complete 22-player windows: {summary["accepted_with_22_observed_players"]}.',
              '', 'The full-match run uses the user-approved positional goalkeeper fallback. YOLO ran independently on 100 frames across four short clips, without a tracker; those detections were not fed into the full-match results.',
              '', 'Bootstrap intervals, class metrics and transition diagnostics are in the per-method JSON files and scoreboard.csv.',
              'The public passage has already been inspected during earlier project development; it is not an untouched test set.']
    lines += ['', '## Sampling uncertainty', '',
              'The following are the official scorer\'s 5th-95th percentile bootstrap intervals (1,000 draws by default).',
              'The PathCRF+rules and best-baseline intervals overlap. The observed gain on 17 possessions does not establish a full-match improvement.', '',
              '| Method | spellF1 bootstrap interval |', '| --- | --- |']
    for _,r in table.iterrows():
        if pd.notna(r['spell_bootstrap_low']):
            lines.append(f'| {r["submission"]} | {r["spell_bootstrap_low"]:.4f} - {r["spell_bootstrap_high"]:.4f} |')
    lines += ['', '[Public possession timeline](public_timeline.png)',
              '[Implementation and reproduction instructions](../../README.md)',
              '[Validation record](VALIDATION.md)']
    (destination/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    valid=np.flatnonzero(truth != pm.X)
    a,b=int(valid[0]),int(valid[-1])+1
    names=['Ground truth']+table['submission'].tolist()
    arrays=[truth[a:b]]+[results[name][a:b] for name in names[1:]]
    fig,ax=plt.subplots(figsize=(14,4.5))
    ax.imshow(np.stack(arrays),interpolation='nearest',aspect='auto',
              cmap=ListedColormap(['#2376b7','#e17c21','#666666','#dddddd']),
              vmin=0,vmax=3,extent=[0,(b-a)/25,len(names)-.5,-.5])
    ax.set_yticks(range(len(names)),names)
    ax.set_xlabel('Seconds from start of the public window')
    ax.set_title('Public-window possession timeline: H blue, A orange, D grey, X light grey')
    fig.tight_layout()
    fig.savefig(destination/'public_timeline.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    main()
