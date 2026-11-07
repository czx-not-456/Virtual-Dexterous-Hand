"""Stage C CAL-only error moments for fair robust6 / robust4 pair.

No 24-frame diagnostic workbook, pinch workbook, or MuJoCo episode labels
are used for fitting the task/NULL gate. Task Jacobian is computed *online*
from the actual simulation; this calibration contains no manufactured J.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.train_tqhg_online import HISTORY
from scripts.stageB_common import historical_split, robust_predictions
from src.retargeting.tqhg_v2 import masked_features


def calibrate(history=HISTORY,v2_file=ROOT/'configs/tqhg_online_v2.json',
              robust6_file=ROOT/'configs/stageB_robust_ridge6.json',
              output=ROOT/'configs/stageC_task_gate.json',seed=20261024,mask_trials=12):
    history,v2_file,robust6_file,output=map(Path,(history,v2_file,robust6_file,output))
    v2=json.loads(v2_file.read_text(encoding='utf-8'))
    r6=json.loads(robust6_file.read_text(encoding='utf-8'))
    if v2.get('schema')!='tqhg_online_calibration_v2' or r6.get('schema')!='stageB_robust_ridge6_v1':
        raise ValueError('StageB/V2 models required')
    if seed != v2['provenance']['seed'] or seed != r6['provenance']['seed']:
        raise ValueError('Seed must be identical to StageB fair training seed')
    sn, y, tr, ca = historical_split(history,v2['base_v1']['model']['sensor_min_deg'],
                                     v2['base_v1']['model']['sensor_span_deg'])
    medians=np.asarray(v2['robust_expert']['sensor_medians_normalized'],float)
    if not np.allclose(medians,r6['sensor_medians_normalized']):
        raise ValueError('Expert input medians differ')
    if mask_trials<2:raise ValueError('Require at least two masks per condition')
    rng=np.random.default_rng(seed+3301)
    errors={}
    for missing in (0,1,3,5):
        ea=[];eb=[]
        for repeat in range(1 if missing==0 else mask_trials):
            for i in ca:
                valid=np.ones(15,bool)
                if missing:
                    valid[rng.choice(15,size=missing,replace=False)]=False
                feature=masked_features(sn[i],valid,medians)
                pred=robust_predictions(feature,v2['base_v1'],v2,r6)
                ea.append(pred['ridge6_robust']-y[i]); eb.append(pred['pca4_robust']-y[i])
        errors[str(missing)]={'a':np.asarray(ea).tolist(),'b':np.asarray(eb).tolist()}
    result={
      'schema':'stageC_task_gate_v1',
      'errors':errors,
      'hyperparameters':{
        'gate_regularizer':0.03,'gate_smoothing':0.2,
        'null_posture_weight':0.025,
      },
      'provenance':{
        'seed':int(seed),'train_count':int(len(tr)),'calibration_count':int(len(ca)),
        'masks_per_fault_stratum':int(mask_trials),
        'residuals':'paired robust6/robust4 normalized control errors on historical CAL ONLY',
        'history_file':str(history),
        'validation_excluded':['captured_samples(1).xlsx','CHM6_pinch_hold_diagnostic_only.xlsx'],
        'simulation_task_projector':'computed from MuJoCo J_tip @ C online; sim=none means P=0',
        'warning':'CAL-only teacher labels; not measured human-hand or grip ground truth'
      },
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[OK] Stage C calibration:',output)
    for missing in (0,1,3,5):
        a,b=np.asarray(errors[str(missing)]['a']),np.asarray(errors[str(missing)]['b'])
        print(f'CAL mask={missing}: records={len(a)} robust6 MAE={np.mean(np.abs(a)):.6f} robust4 MAE={np.mean(np.abs(b)):.6f}')
    print('[NOTE] No external test data were used. Simulator Jacobian is evaluated at each runtime state.')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--history',type=Path,default=HISTORY)
    ap.add_argument('--v2-model',type=Path,default=ROOT/'configs/tqhg_online_v2.json')
    ap.add_argument('--robust6-model',type=Path,default=ROOT/'configs/stageB_robust_ridge6.json')
    ap.add_argument('--output',type=Path,default=ROOT/'configs/stageC_task_gate.json')
    ap.add_argument('--seed',type=int,default=20261024)
    ap.add_argument('--mask-trials',type=int,default=12)
    a=ap.parse_args()
    calibrate(a.history,a.v2_model,a.robust6_model,a.output,a.seed,a.mask_trials)
