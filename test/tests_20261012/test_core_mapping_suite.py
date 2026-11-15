"""Pure-code QA for the pruned mapping experiment suite (no MuJoCo required)."""
from pathlib import Path

import numpy as np
import pytest

from experiments.core_mapping_evidence import pose_paired_bootstrap, read_run_rows
from experiments.run_core_mapping_suite import plan_commands


def test_pose_bootstrap_does_not_treat_frames_as_independent_subjects():
    diffs = np.array([1., 1., 1., 1., -1., -1.])
    stats = pose_paired_bootstrap(diffs, ['A']*4 + ['B']*2, repeats=120, seed=4)
    assert stats['pose_cluster_count'] == 2
    assert stats['n_paired_frames'] == 6
    assert stats['mean_paired_delta_deg'] == pytest.approx(0.)
    assert stats['ci95_pose_cluster_bootstrap'] is not None
    assert pose_paired_bootstrap(diffs, ['A']*4+['B']*2, repeats=120, seed=4) == stats


def test_single_pose_ci_is_unavailable_not_fabricated():
    stats = pose_paired_bootstrap(np.array([1.,2.]), ['same','same'], repeats=100)
    assert stats['ci95_pose_cluster_bootstrap'] is None
    with pytest.raises(ValueError):
        pose_paired_bootstrap(np.array([float('nan')]),['bad'])


def test_missing_or_misaligned_rows_are_rejected(tmp_path):
    f=tmp_path/'run.csv'
    f.write_text('frame,q_th_cmc_joint\n0,0\n2,1\n',encoding='utf-8')
    with pytest.raises(ValueError,match='aligned'):
        read_run_rows(f,2)
    with pytest.raises(ValueError,match='Expected'):
        read_run_rows(f,5)
    with pytest.raises(FileNotFoundError):
        read_run_rows(tmp_path/'missing.csv',2)


def test_suite_only_executes_retained_mapping_comparisons(tmp_path):
    commands=plan_commands('python',tmp_path/'sample.xlsx',tmp_path,
                           {'stageb','modules','spectrum'},24,'sensor',
                           seed=7,bootstrap=100,mask_repeats=2)
    assert [name for name,_ in commands] == ['StageB','B','C','D','E']
    e=commands[-1][1]
    assert '--modes' in e
    assert 'scheme_E' in e[-1] and 'E_fixed6' in e[-1]
    assert 'E_full6_feasible' not in e[-1]
    assert not any('compare_mapping' in ' '.join(cmd) or 'glove_calibration' in ' '.join(cmd)
                   for _,cmd in commands)
