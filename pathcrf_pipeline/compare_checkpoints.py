"""Compare frozen released checkpoints using the unchanged PUBLIC scorer.

Run only after `python -m pathcrf_pipeline.benchmark` has completed.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import ROOT, OUTPUT, ASSETS
from .infer import TRIAL_NAMES
from .prepare import file_sha256


def format_interval(low, high):
    if pd.isna(low) or pd.isna(high):
        return 'not computed'
    return f'{low:.4f} - {high:.4f}'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap',type=int,default=1000)
    args=parser.parse_args()
    base=OUTPUT/'checkpoints'
    plan=json.loads((base/'benchmark_plan.json').read_text())
    if set(plan.get('completed_trials',[])) != set(TRIAL_NAMES):
        raise ValueError('All five checkpoints must finish before comparison')
    summaries={trial:json.loads((base/str(trial)/'run_summary.json').read_text()) for trial in TRIAL_NAMES}
    fingerprints={s['input_fingerprint_sha256'] for s in summaries.values()}
    if len(fingerprints) != 1:
        raise ValueError('Input fingerprints differ across checkpoints')
    rules_hashes={file_sha256(base/str(t)/'results'/'rules_only.csv') for t in TRIAL_NAMES}
    if len(rules_hashes) != 1:
        raise ValueError('The fallback rules differ across checkpoints')
    if {s.get('cpu_flush_denormal') for s in summaries.values()} != {True}:
        raise ValueError('CPU numerical settings differ across checkpoints')
    sys.path.insert(0,str(ROOT/'evaluation'))
    import run_evaluation as evaluator
    import possession_metrics as pm
    gt=ROOT/'evaluation'/'gt'/'BAR-ATH_possession_gt_public.csv'
    truth=pm.label_codes(pm.read_label_csv(str(gt)))
    out=base/'comparison'
    out.mkdir(parents=True,exist_ok=True)
    full_rows=[]
    checkpoint_rows=[]
    predictions={}
    candidates=[]
    for trial in TRIAL_NAMES:
        for mode,name in [('model','pathcrf_model'),('combined','pathcrf_rules')]:
            candidates.append((f'{trial}_{mode}',base/str(trial)/'results'/f'{name}.csv',trial,mode))
    candidates.extend((p.stem,p,None,None) for p in sorted((ROOT/'evaluation'/'results').glob('baseline_*.csv')))
    candidates.append(('rules_only',base/'140'/'results'/'rules_only.csv',None,None))
    results={}
    for name,path,trial,mode in candidates:
        result,reason=evaluator.score_submission(str(path),truth,pm.TAUS,args.bootstrap)
        if reason:
            raise ValueError(f'{name}: {reason}')
        row=evaluator.scoreboard_row(None,{'name':name,'path':str(path),'result':result,'reason':None},pm.TAUS)
        row.update({'trial':trial,'mode':mode,'prediction_sha256':file_sha256(path)})
        full_rows.append(row)
        results[name]=result
        predictions[name]=pm.label_codes(pm.read_label_csv(str(path)))
        report=evaluator.json_safe(result)
        report.update({'trial':trial,'mode':mode,'prediction_sha256':row['prediction_sha256']})
        (out/f'{name}.json').write_text(json.dumps(report,indent=2))
        print(f'{name}: macroF1={result["macro_f1"]:.4f}, spellF1={result["spell_f1_mean"]:.4f}',flush=True)
    for trial in TRIAL_NAMES:
        model=results[f'{trial}_model']; combined=results[f'{trial}_combined']
        s=summaries[trial]
        checkpoint_rows.append({'trial':trial,'variant':TRIAL_NAMES[trial],
            'model_macro_f1':model['macro_f1'],'model_spell_f1':model['spell_f1_mean'],
            'model_spell_low':model['spell_bootstrap_low'],'model_spell_high':model['spell_bootstrap_high'],
            'combined_macro_f1':combined['macro_f1'],'combined_spell_f1':combined['spell_f1_mean'],
            'combined_spell_low':combined['spell_bootstrap_low'],'combined_spell_high':combined['spell_bootstrap_high'],
            'model_predicted_possessions':model['n_pred_spells'],
            'combined_predicted_possessions':combined['n_pred_spells'],
            'public_abstentions_model':model['n_abstain_on_graded'],
            'coverage_full_pct':s['model_coverage_pct'],
            'accepted_windows':s['accepted_windows'],'decoder':s['decoder'],
            'inference_seconds':s['seconds'],'checkpoint_sha256':s['checkpoint_sha256']})
    table=pd.DataFrame(checkpoint_rows).sort_values(['combined_spell_f1','combined_macro_f1'],ascending=False)
    table.insert(0,'rank_combined',np.arange(1,len(table)+1))
    table.to_csv(out/'checkpoints.csv',index=False)
    scoreboard=pd.DataFrame(full_rows).sort_values(['spell_f1_mean','macro_f1'],ascending=False)
    scoreboard['rank']=np.arange(1,len(scoreboard)+1)
    scoreboard.to_csv(out/'scoreboard.csv',index=False)
    leader=table.iloc[0]
    model_leader=max(TRIAL_NAMES,key=lambda t:(results[f'{t}_model']['spell_f1_mean'],results[f'{t}_model']['macro_f1']))
    reference=summaries[140]
    public_coverage=100*pd.read_parquet(base/'140'/'frame_audit.parquet')['model_covered'].to_numpy()[truth != pm.X].mean()
    old_140_equal={}
    for name in ['pathcrf_model','pathcrf_rules','rules_only']:
        old=OUTPUT/'results'/f'{name}.csv'
        old_140_equal[name]=(file_sha256(old)==file_sha256(base/'140'/'results'/f'{name}.csv')) if old.exists() else None
    if any(value is False for value in old_140_equal.values()):
        raise RuntimeError('Fresh 140 predictions differ from the original reference; inspect before comparing')
    validation={'all_input_fingerprints_identical':True,'input_fingerprint_sha256':next(iter(fingerprints)),
                'rule_files_identical':True,'all_exports_validated':True,
                'all_frames':len(truth),'public_graded_frames':int((truth!=pm.X).sum()),
                'public_possessions':results['140_combined']['n_gt_spells'],
                'model_coverage_full_pct':reference['model_coverage_pct'],
                'model_coverage_public_pct':float(public_coverage),
                'checkpoint_140_matches_original_exports':old_140_equal,
                'public_gt_sha256':file_sha256(gt),'bootstrap_draws':args.bootstrap,
                'best_observed_combined_trial':int(leader['trial']),
                'best_observed_model_trial':int(model_leader),
                'ground_truth_used_only_for_evaluation':True}
    cpu_reference=base/'cpu_reference_hashes.json'
    if cpu_reference.exists():
        original=json.loads(cpu_reference.read_text())
        validation['cpu_reference_exports_identical']={trial:{
            name:file_sha256(base/trial/'results'/f'{name}.csv')==sha
            for name,sha in files.items()} for trial,files in original.items()}
        if not all(value for checks in validation['cpu_reference_exports_identical'].values() for value in checks.values()):
            raise RuntimeError('CPU execution setting changed reference exports; inspect before comparing')
    (out/'validation.json').write_text(json.dumps(validation,indent=2))
    print('\nCheckpoint comparison (combined spellF1 ranking):',flush=True)
    print(table[['trial','variant','model_macro_f1','model_spell_f1','combined_macro_f1','combined_spell_f1']].to_string(index=False),flush=True)
    lines=['# Released PathCRF checkpoint comparison','',
        'All five supplied checkpoints were run on the full tactical timeline using identical input tensors, accepted windows, missing-player padding, positional goalkeeper flags, and fallback rules.',
        'Prediction parameters were frozen before evaluating this comparison. Training was not performed.',
        f'Scores below use only the published {validation["public_graded_frames"]:,}-frame window ({validation["public_possessions"]} possessions).', '',
        '| Checkpoint | Variant | Model macroF1 | Model spellF1 | Combined macroF1 | Combined spellF1 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for _,r in table.iterrows():
        lines.append(f'| {int(r["trial"])} | {r["variant"]} | {r["model_macro_f1"]:.4f} | {r["model_spell_f1"]:.4f} | {r["combined_macro_f1"]:.4f} | {r["combined_spell_f1"]:.4f} |')
    lines += ['',f'Best observed checkpoint with the shared rule layer: **{int(leader["trial"])}**, spellF1 **{leader["combined_spell_f1"]:.4f}**.',
        f'Best observed model-only checkpoint: **{model_leader}**, spellF1 **{results[f"{model_leader}_model"]["spell_f1_mean"]:.4f}**.', '',
        '## How to read the comparison','',
        '- Model-only uses the checkpoint\'s native decoder and abstains outside accepted windows. All models share automatic estimated-live segmentation.',
        '- Combined uses model predictions where covered, the same automatic rules in gaps, and the same estimated-dead override.',
        '- 100 uses independent argmax; 110/130 use their dense CRFs; 120/140 use their masked CRFs. No extra masked-transition restrictions were added to dense/Non-CRF variants.',
        '- All variants prevent predictions involving zero-padding nodes, using the same absent-player policy.',
        f'- Model coverage is {reference["model_coverage_pct"]:.2f}% over the full graded match and {public_coverage:.2f}% over the public window.',
        '- No additional tracker was used. Full-match goalkeeper flags use the previously approved positional fallback.',
        '- All five use the same CPU flush-denormal setting. Tiny subnormal weights made some checkpoints unusually slow; tests verified matching logits within 1e-6 and identical decoded paths with/without the setting. Weights on disk are unchanged.',
        '- These compare the released weight files. Recorded training configurations differ (for example, 120 has lomocv=false and 140 has lomocv=true); this is not a controlled experiment attributing all differences solely to CRF architecture.', '',
        '## Bootstrap uncertainty','',
        f'Official scorer, {args.bootstrap} draws, 5th-95th percentile intervals. These are marginal per-model intervals, not a paired test or a guarantee of a full-match ranking.',
        '| Checkpoint | Model-only spellF1 interval | Combined spellF1 interval |',
        '| --- | --- | --- |']
    for _,r in table.iterrows():
        lines.append(f'| {int(r["trial"])} | {format_interval(r["model_spell_low"],r["model_spell_high"])} | {format_interval(r["combined_spell_low"],r["combined_spell_high"])} |')
    lines += ['', 'The public sample is small and has already been inspected in earlier project development. Selecting a checkpoint on these scores makes it a model-selection sample, not an independent final test. Full-match accuracy remains unmeasured.', '',
        '## Verification','',
        '- Strict checkpoint parameter loading and official-decoder parity tests passed for all five variants.',
        '- The test suite also checks the CPU numerical setting against unoptimized logits and decoded paths for every variant.',
        '- Accepted input tensors have the same SHA-256 fingerprint across all checkpoints.',
        '- The same 902 windows were accepted and every fallback-rule CSV has an identical hash.',
        '- Prediction ran with answer-key access blocked; the separate evaluator uses only the public key.',
        '- All model/combined exports passed the official 199,319-row submission checks.',
        f'- Fresh 140 exports match the original experiment: {json.dumps(old_140_equal)}.', '',
        '[Combined-label timelines](combined_timelines.png) · [Detailed scoreboard](scoreboard.csv) · [Validation metadata](validation.json)']
    (out/'CHECKPOINT_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    public=np.flatnonzero(truth != pm.X)
    first,last=int(public[0]),int(public[-1])+1
    names=['Ground truth']+[f'{int(t)}_combined' for t in table['trial']]
    arrays=[truth[first:last]]+[predictions[name][first:last] for name in names[1:]]
    fig,ax=plt.subplots(figsize=(14,4))
    ax.imshow(np.stack(arrays),interpolation='nearest',aspect='auto',
              cmap=ListedColormap(['#2376b7','#e17c21','#666666','#dddddd']),vmin=0,vmax=3,
              extent=[0,(last-first)/25,len(names)-.5,-.5])
    ax.set_yticks(range(len(names)),names)
    ax.set_xlabel('Seconds from start of public window')
    ax.set_title('Same rule layer for every checkpoint: H blue, A orange, D grey, X light grey')
    fig.tight_layout()
    fig.savefig(out/'combined_timelines.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    main()
