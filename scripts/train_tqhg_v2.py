"""Fit TQHG-v2 on historical TRAIN split; gate on disjoint CALIBRATION.

No reading of 24-frame external test data or pinch diagnostic workbook.
Synthetic failure masks are not proof of physical sensor-fault robustness.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.retargeting.tqhg_v2 import masked_features, solve_ridge
from src.retargeting.latent_synergy import CONTROL_LIMITS_DEG
from scripts.train_tqhg_online import HISTORY

DEFAULT = ROOT / 'configs/tqhg_online_v2.json'
BASE = ROOT / 'configs/tqhg_online_v1.json'


def augment(normalized, medians, rng, repeat=5):
    """Mask/noise/dropout using only training samples; no external test access."""
    n=len(normalized)
    observations=[]
    masks=[]
    owners=[]
    for trial in range(repeat+1):
        noisy=normalized.copy()
        valid=np.ones((n,15),bool)
        if trial > 0:
            # Fault locations are not fixed across samples.
            number = rng.choice([0,1,2,3,5],size=n,p=[.10,.25,.20,.35,.10])
            for i,k in enumerate(number):
                if k:
                    channels=rng.choice(15,size=k,replace=False)
                    valid[i,channels]=False
            noisy += rng.normal(0.,rng.choice([.008,.025,.06]),size=noisy.shape)
            if trial%2 == 0:
                noisy += rng.normal(0.,.035,size=(n,15))
        observations.append(noisy)
        masks.append(valid)
        owners.append(np.arange(n))
    return masked_features(np.vstack(observations),np.vstack(masks),medians),np.hstack(owners)


def synthetic_fault_batch(sn, medians, rng, trials=3):
    """Independent RNG from training aug, fault scenarios for CAL split."""
    all_x=[]
    for trial in range(trials):
        corrupt=sn.copy()
        mask=np.ones(sn.shape,bool)
        number=1 if trial==0 else 3 if trial==1 else 2
        for i in range(len(sn)):
            mask[i,rng.choice(15,size=number,replace=False)]=False
        corrupt += rng.normal(0.,.025 if trial!=2 else .045,size=sn.shape)
        all_x.append(masked_features(corrupt,mask,medians))
    return np.vstack(all_x)


def train(history=HISTORY, base_file=BASE, output=DEFAULT, seed=20261024):
    history, base_file, output=Path(history),Path(base_file),Path(output)
    base=json.loads(base_file.read_text(encoding='utf-8'))
    if base.get('schema')!='tqhg_online_calibration_v1':
        raise ValueError('Need original v1 calibration JSON')
    with history.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    sensors=np.asarray([[float(r[f'sensor_{j}']) for j in range(1,16)] for r in rows])
    labels=np.asarray([[float(r[f'target_{j}']) for j in range(1,7)] for r in rows])/CONTROL_LIMITS_DEG
    poses=[r['pose'] for r in rows]
    tr=[];ca=[]
    for pose in dict.fromkeys(poses):
        ix=[i for i,p in enumerate(poses) if p==pose]
        n=max(2,min(len(ix)-2,int(np.floor(len(ix)*.66))))
        tr.extend(ix[:n]);ca.extend(ix[n:])
    tr=np.asarray(tr,dtype=int); ca=np.asarray(ca,dtype=int)
    p=base['model']
    smin,span=np.asarray(p['sensor_min_deg']),np.asarray(p['sensor_span_deg'])
    sn=np.clip((sensors-smin)/span,0.,1.)
    medians=np.median(sn[tr],axis=0)
    rng=np.random.default_rng(seed)
    X,owners=augment(sn[tr],medians,rng)
    target_latent=(labels[tr]-np.asarray(p['control_mean_normalized'])) @ np.asarray(p['control_pca_basis'])
    W,b=solve_ridge(X,target_latent[owners],regularizer=2.0)
    mu=np.asarray(p['control_mean_normalized'])
    basis=np.asarray(p['control_pca_basis'])
    Wr=np.asarray(p['sensor_to_controls_weights'])
    br=np.asarray(p['sensor_to_controls_bias'])

    def predictions(features):
        ridge=np.clip(features[:,:15]@Wr+br,0.,1.)
        robust=np.clip(mu+(features@W+b)@basis.T,0.,1.)
        return ridge,robust
    xclean=masked_features(sn[ca],np.ones((len(ca),15),bool),medians)
    xfault=synthetic_fault_batch(sn[ca],medians,np.random.default_rng(seed+99))
    a0,b0=predictions(xclean)
    a1,b1=predictions(xfault)
    yclean=labels[ca]
    yfault=np.tile(yclean,(3,1))
    result={
        'schema':'tqhg_online_calibration_v2',
        'base_v1':base,
        'robust_expert':{
            'feature_to_latent_weights':W.tolist(),
            'feature_to_latent_bias':b.tolist(),
            'sensor_medians_normalized':medians.tolist(),
            'input_description':'15 normalized imputed values + 15 observed-channel masks; 4D PCA latent output',
        },
        'gate_calibration':{
            'normal_errors_ridge':(a0-yclean).tolist(),
            'normal_errors_robust':(b0-yclean).tolist(),
            'fault_errors_ridge':(a1-yfault).tolist(),
            'fault_errors_robust':(b1-yfault).tolist(),
        },
        'hyperparameters':{
            'gate_regularizer':0.03,
            'gate_smoothing':0.2,
            'null_posture_weight':0.025,
            'null_smooth_weight':0.015,
        },
        'provenance':{
            'history':str(history),
            'v1_model_source':str(base_file),
            'train_count':int(len(tr)),
            'calibration_count':int(len(ca)),
            'seed':int(seed),
            'split':'chronological first ~66% of each historical gesture for training; rest calibration',
            'synthetic_fault_trials':3,
            'training_augmentations':'mask 1-5 channels; Gaussian jitter; no external test samples',
            'teacher_labels_only':True,
            'test_files_excluded':['captured_samples(1).xlsx','CHM6_pinch_hold_diagnostic_only.xlsx'],
            'warning':'Small six-pose single-device proxy labels: proof of concept, not validation of real fault robustness.',
        },
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    clean_r=float(np.mean(np.abs(a0-yclean)))
    clean_b=float(np.mean(np.abs(b0-yclean)))
    fault_r=float(np.mean(np.abs(a1-yfault)))
    fault_b=float(np.mean(np.abs(b1-yfault)))
    print(f'[OK] TQHG-v2 trained: {output}')
    print(f'[DATA] train={len(tr)} calibration={len(ca)} separate; synthetic CAL fault cases={len(yfault)}')
    print(f'[CAL proxy MAE normalized] normal Ridge={clean_r:.6f}, Robust={clean_b:.6f}')
    print(f'[CAL proxy MAE normalized] degraded Ridge={fault_r:.6f}, Robust={fault_b:.6f}')
    print('[NOTE] CAL metrics are not independent final test results; do not claim robust superiority without held-out tests.')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--history',type=Path,default=HISTORY)
    ap.add_argument('--base',type=Path,default=BASE)
    ap.add_argument('--output',type=Path,default=DEFAULT)
    ap.add_argument('--seed',type=int,default=20261024)
    args=ap.parse_args()
    train(args.history,args.base,args.output,args.seed)
