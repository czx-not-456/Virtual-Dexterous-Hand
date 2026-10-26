"""Fairness ablation: fit 30-input robust 6D Ridge using EXACT v2 augmentation.

No test workbooks are read. New model is a separate Stage B file; never
modifies/overwrites the original v1/v2 models or the MuJoCo controller.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_tqhg_online import HISTORY
from scripts.train_tqhg_v2 import augment
from scripts.stageB_common import historical_split, CONDITIONS, robust_predictions, scalar_calibration_weight
from src.retargeting.tqhg_v2 import masked_features,solve_ridge


def train(history=HISTORY,v2_file=ROOT/'configs/tqhg_online_v2.json',
          output=ROOT/'configs/stageB_robust_ridge6.json',seed=20261024,
          regularizer=2.0,calibration_masks=12):
    history,v2_file,output=Path(history),Path(v2_file),Path(output)
    if output.resolve() in {v2_file.resolve(), (ROOT/'configs/tqhg_online_v1.json').resolve()}:
        raise ValueError('Output must not overwrite TQHG-v1 or TQHG-v2 model')
    cfg=json.loads(v2_file.read_text(encoding='utf-8'))
    if cfg.get('schema')!='tqhg_online_calibration_v2':
        raise ValueError('Expected TQHG-v2 model file')
    v1=cfg['base_v1'];p=v1['model']; provenance=cfg['provenance']
    if provenance['train_count']!=124 or provenance['calibration_count']!=66:
        raise ValueError('TQHG-v2 uses incompatible historical split')
    if int(provenance.get('seed',-1))!=seed:
        raise ValueError('Training seed must match TQHG-v2 to use the same augmented observations')
    sn,y,tr,ca=historical_split(history,p['sensor_min_deg'],p['sensor_span_deg'])
    medians=np.median(sn[tr],axis=0)
    cfg_medians=np.asarray(cfg['robust_expert']['sensor_medians_normalized'],float)
    if not np.allclose(medians,cfg_medians,atol=1e-10):
        raise ValueError('Historical data/normalization does not match existing v2 robust expert')
    x,owner=augment(sn[tr],medians,np.random.default_rng(seed),repeat=5)
    weights,bias=solve_ridge(x,y[tr][owner],regularizer=regularizer)
    if weights.shape!=(30,6) or not np.isfinite(weights).all():
        raise ValueError('Robust6 fit failed')
    model={'schema':'stageB_robust_ridge6_v1',
           'source_tqhg_v2':'configs/tqhg_online_v2.json',
           'feature_to_controls_weights':weights.tolist(),
           'feature_to_controls_bias':bias.tolist(),
           'sensor_medians_normalized':medians.tolist(),
           'calibrated_scalar_weights':{},
           'provenance':{'historical_training_file':'数据手套数据/records/chm6_calibrations/20260917_102742_G12LC040704_left/captured_samples.csv',
                         'train_count':124,'calibration_count':66,
                         'seed':seed,'training_augments_repeat':5,
                         'regularizer':regularizer,
                         'shared_augmentation_with_v2_robust4':True,
                         'feature_dimensions':30,'target_dimensions':6,
                         'calibration_masks_per_condition':calibration_masks,
                         'evaluation_set_excluded':['captured_samples(1).xlsx','CHM6_pinch_hold_diagnostic_only.xlsx'],
                         'labels':'normalized software-teacher control labels, not human ground truth'}}
    rng=np.random.default_rng(seed+901)
    from scripts.stageB_common import robust_predictions
    for condition,n_missing in CONDITIONS:
        ea,eb=[],[]
        for _ in range(1 if n_missing==0 else calibration_masks):
            for i in ca:
                mask=np.ones(15,bool)
                if n_missing:mask[rng.choice(15,size=n_missing,replace=False)]=False
                features=masked_features(sn[i],mask,medians)
                out=robust_predictions(features,v1,cfg,model)
                ea.append(out['ridge6_robust']-y[i])
                eb.append(out['pca4_robust']-y[i])
        model['calibrated_scalar_weights'][condition]=scalar_calibration_weight(np.stack(ea),np.stack(eb))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(model,indent=2,ensure_ascii=False),encoding='utf-8')
    print('[OK] Stage B augmented Ridge6 trained:', output)
    print('[DATA] historical training=124 calibration=66; augmentation rows=',len(x))
    print('[CAL ONLY] scalar weights:', ', '.join(f'{k}={v:.4f}' for k,v in model['calibrated_scalar_weights'].items()))
    print('[NOTE] No 24-frame independent test labels used in fitting. This is offline comparison, not MuJoCo.')
    return model

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--history',type=Path,default=HISTORY)
    ap.add_argument('--v2-model',type=Path,default=ROOT/'configs/tqhg_online_v2.json')
    ap.add_argument('--output',type=Path,default=ROOT/'configs/stageB_robust_ridge6.json')
    ap.add_argument('--seed',type=int,default=20261024)
    ap.add_argument('--regularizer',type=float,default=2.)
    a=ap.parse_args()
    train(a.history,a.v2_model,a.output,a.seed,a.regularizer)
