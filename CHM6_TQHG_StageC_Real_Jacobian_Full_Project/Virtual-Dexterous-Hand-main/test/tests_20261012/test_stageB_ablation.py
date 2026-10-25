"""Stage B unit and integration smoke tests (no MuJoCo required)."""
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.stageB_common import paired_cases,robust_predictions,predict_blends,scalar_calibration_weight
from scripts.evaluate_stageB_ablation import pose_cluster_ci
from scripts.train_stageB_robust_ridge6 import train


def test_weight_selects_b_when_b_superior():
    a=np.full((15,6),.25); b=np.zeros((15,6))
    assert scalar_calibration_weight(a,b)>.9


def test_weight_selects_a_when_b_worse():
    a=np.zeros((15,6));b=np.full((15,6),.25)
    assert scalar_calibration_weight(a,b)<.02


def test_paired_mask_repeat_determinism_and_clean():
    seq1=list(paired_cases(np.zeros((3,15)),np.random.default_rng(24),4))
    seq2=list(paired_cases(np.zeros((3,15)),np.random.default_rng(24),4))
    assert len(seq1)==3*(1+3*4)
    assert [tuple(x[4]) for x in seq1]==[tuple(x[4]) for x in seq2]
    assert [np.sum(~x[4]) for x in seq1[:3]]==[0]*3
    assert set(np.sum(~x[4]) for x in seq1)=={0,1,3,5}


def test_pose_cluster_bootstrap_not_case_level():
    data=[{'pose':f'p{j}','A':0.1*j,'B':0.} for j in range(6) for _ in range(20)]
    low,high=pose_cluster_ci(data,'A','B',np.random.default_rng(7),500)
    assert 0<=low<=.25<=high<=.5


def test_training_safely_separates_history_from_test(tmp_path):
    p=Path('configs/tqhg_online_v2.json')
    if not p.exists():pytest.skip('TQHG-v2 model not available')
    dest=tmp_path/'robust6.json'
    model=train(output=dest)
    assert dest.exists()
    assert np.asarray(model['feature_to_controls_weights']).shape==(30,6)
    assert model['provenance']['train_count']==124
    assert model['provenance']['calibration_count']==66
    assert all(0<=w<=1 for w in model['calibrated_scalar_weights'].values())
    v2=json.loads(p.read_text(encoding='utf-8'))
    x=np.concatenate((np.full(15,.42),np.ones(15)))
    d=predict_blends(x,v2['base_v1'],v2,model,'clean')
    assert len(d)==6
    assert all(val.shape==(6,) and np.isfinite(val).all() for val in d.values())
    assert np.allclose(d['fixed_50_50'],.5*(d['pca4_robust']+d['ridge6_robust']))


def test_training_rejects_wrong_seed(tmp_path):
    with pytest.raises(ValueError,match='seed'):
        train(output=tmp_path/'wrong.json',seed=7)


def test_end_to_end_evaluate_creates_paired_outputs(tmp_path):
    from scripts.evaluate_stageB_ablation import evaluate
    path=Path('configs/stageB_robust_ridge6.json')
    if not path.exists():pytest.skip('Run Stage B trainer first')
    d=tmp_path/'report'
    result=evaluate(Path('数据手套数据/captured_samples(1).xlsx'),
                    Path('configs/tqhg_online_v2.json'),path,d,
                    trials=1,seed=20261025,bootstrap=120)
    assert result['n_external_records']==24
    assert len(result['summary'])==4
    assert all(r['paired_cases']==24 for r in result['summary'])
    assert (d/'paired_cases.csv').exists() and (d/'summary_by_pose.csv').exists()
    assert (d/'REPORT.md').exists() and (d/'summary.json').exists()
    for r in result['summary']:
        assert r['ridge6_robust']>=0 and r['pca4_robust']>=0
