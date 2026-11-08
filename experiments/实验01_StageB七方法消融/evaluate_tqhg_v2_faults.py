"""Held-out 24-frame external glove test, synthetic missing-channel comparison.

Only used AFTER training/calibration. This script never updates any weights.
The 24 records are six fixed poses, not 24 independent dynamic gestures.
When sim is absent task projector is unavailable, so this does NOT evaluate
MuJoCo task-facing fingertip accuracy or stable grasp.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.glove.captured_samples import load_captured_samples
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.latent_synergy import CONTROL_LIMITS_DEG
from src.retargeting.tqhg_v2 import TQHGv2OnlineMapper, masked_features


def evaluate(test_path, model_path, seed=20261025, trials=20):
    captured=load_captured_samples(test_path)
    if any(item.targets_deg is None for item in captured):
        raise ValueError('The independent test file must have all 6 teacher target labels')
    model=TQHGv2OnlineMapper(RobotHandModel(), model_path)
    sensors=np.asarray([x.sensors_deg for x in captured])
    refs=np.asarray([x.targets_deg for x in captured])/CONTROL_LIMITS_DEG
    gen=np.random.default_rng(seed)
    W,b=model.W_ridge,model.b_ridge
    print(f'INDEPENDENT TEST: {test_path}, samples={len(captured)}, trials={trials}; seed={seed}')
    print('MAE normalized 6D control units; explicit fault mask supplied equally to experts; not actual robot truth.')
    print('condition\tridge6\trobust4\ttqhg-v2\tgate_null_mean')
    for label,number in [('clean',0),('mask_one',1),('mask_three',3),('mask_five',5)]:
        estimates=[]
        for trial in range(trials if number else 1):
            maer=[];maeb=[];maem=[];wn=[]
            for s,y in zip(sensors,refs):
                valid=np.ones(15,bool)
                if number:
                    valid[gen.choice(15,size=number,replace=False)]=False
                model.reset()
                sn, effective_mask, _, _ = model.quality.observe(s,valid)
                features=masked_features(sn,effective_mask,model.medians)
                r=np.clip(features[:15]@W+b,model.lower,model.upper)
                rob=np.clip(model.mu+(features@model.robust_W+model.robust_b)@model.robust_basis.T,
                            model.lower,model.upper)
                model.reset()
                pred=model.predict(s,method='tqhg-v2',sensor_valid_mask=valid,simulator=None)
                maer.append(np.mean(np.abs(r-y)))
                maeb.append(np.mean(np.abs(rob-y)))
                maem.append(np.mean(np.abs(pred-y)))
                wn.append(model.last_gate['w_null'])
            estimates.append([np.mean(x) for x in (maer,maeb,maem,wn)])
        q=np.mean(estimates,axis=0)
        print(f'{label}\t{q[0]:.6f}\t{q[1]:.6f}\t{q[2]:.6f}\t{q[3]:.4f}')
    print('WARNING: Test poses are few; synthetic corruption tests robustness of teacher-label reconstruction only.')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--test',type=Path,default=ROOT/'数据手套数据/captured_samples(1).xlsx')
    ap.add_argument('--model',type=Path,default=ROOT/'configs/tqhg_online_v2.json')
    ap.add_argument('--seed',type=int,default=20261025)
    ap.add_argument('--trials',type=int,default=20)
    a=ap.parse_args()
    evaluate(a.test,a.model,a.seed,a.trials)
