"""Produce conservative, auditable pose-paired reports for offline mapping experiments.

Not a significance test, not measured robot ground truth, not a physical grasp metric.
The six preset poses are the resampling clusters; repeated frames are not subjects.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def hash_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            sha.update(block)
    return sha.hexdigest()


def pose_paired_bootstrap(differences: np.ndarray, poses: list[str], *, seed: int = 20261108,
                          repeats: int = 3000) -> dict[str, Any]:
    """Equal-weight each pose, resample whole pose clusters (descriptive only)."""
    diff = np.asarray(differences, dtype=float)
    if diff.ndim != 1 or len(diff) != len(poses) or not len(diff) or not np.isfinite(diff).all():
        raise ValueError('Paired errors must be finite and match pose labels')
    if repeats < 100:
        raise ValueError('Need >= 100 bootstrap draws')
    by_pose: dict[str, list[float]] = defaultdict(list)
    for pose, item in zip(poses, diff):
        by_pose[str(pose)].append(float(item))
    groups = sorted(by_pose)
    means = np.array([np.mean(by_pose[group]) for group in groups], dtype=float)
    result = {'pose_cluster_count':len(groups), 'n_paired_frames':len(diff),
              'mean_paired_delta_deg':float(means.mean()),
              'pose_means_delta_deg':{p:float(m) for p,m in zip(groups, means)},
              'ci95_pose_cluster_bootstrap':None,
              'bootstrap_note':'exploratory; few fixed pose classes, not independent participants'}
    if len(groups)>1:
        indices = np.random.default_rng(seed).integers(0,len(means),size=(repeats,len(means)))
        lo,hi=np.quantile(np.mean(means[indices],axis=1),[.025,.975])
        result['ci95_pose_cluster_bootstrap']=[float(lo),float(hi)]
    return result


def read_run_rows(path: Path, expected_frames: int) -> list[dict[str,str]]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing freshly generated run: {path}')
    with path.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    if len(rows)!=expected_frames:
        raise ValueError(f'Expected {expected_frames} rows, got {len(rows)} in {path}')
    frames=[int(float(row['frame'])) for row in rows]
    if frames != list(range(expected_frames)):
        raise ValueError(f'Frames must be aligned 0..{expected_frames-1} (no loops/drops): {path}')
    return rows


def get_error_and_timing(rows, *, joint_names, active_indices, labels_deg):
    q=np.asarray([[float(row['q_'+name]) for name in joint_names] for row in rows],dtype=float)
    if not np.isfinite(q).all():
        raise ValueError('Nonfinite joint output')
    controls=np.rad2deg(q[:,active_indices])
    if controls.shape != labels_deg.shape:
        raise ValueError(f'Teacher label shape mismatch: {controls.shape} / {labels_deg.shape}')
    err=np.abs(controls-labels_deg).mean(axis=1)
    optimizer=np.array([float(row['optimizer_ms']) for row in rows])
    latency=np.array([float(row['pipeline_latency_ms']) for row in rows])
    if not (np.isfinite(err).all() and np.isfinite(optimizer).all() and np.isfinite(latency).all()):
        raise ValueError('Nonfinite error/timing metrics')
    if np.any(optimizer < 0) or np.any(latency < 0):
        raise ValueError('Invalid negative timing')
    return err, optimizer, latency


COMPARISONS={
    'modules': [
        ('B:adaptive_B minus fixed_PCA_k4','B', 'diagnostic_adaptive_B','diagnostic_fixed_PCA_k4'),
        ('C:reachability_C minus adaptive_B','C', 'diagnostic_reachability_C','diagnostic_adaptive_B'),
        ('D:credible_D minus reachability_C','D', 'credible_D_diagnostic','reachability_C'),
        ('D:without_uncertainty minus D','D', 'D_without_uncertainty','credible_D_diagnostic'),
        ('D:without_nonlinear_guard minus D','D', 'D_without_nonlinear_guard','credible_D_diagnostic'),
    ],
    'spectrum': [
        ('E:dynamic minus E_fixed6','E', 'scheme_E','E_fixed6'),
        ('E:dynamic minus direct_11DoF','E', 'scheme_E','original_direct'),
    ]
}


def build_evidence(root: Path, samples_file: Path, groups: set[str], *, steps: int, seed: int = 20261108,
                   bootstrap: int = 3000) -> dict[str,Any]:
    """Analyze run artifacts from the requested groups only; never invent missing results."""
    from src.glove.captured_samples import load_captured_samples
    from src.hand_model.robot_hand import RobotHandModel
    from src.retargeting.latent_synergy import LatentSynergyMapper

    root=Path(root).resolve();samples_file=Path(samples_file).resolve()
    captured=load_captured_samples(samples_file)
    if steps < 1 or steps > len(captured):
        raise ValueError('steps must be 1..dataset_length; no repeated samples')
    used=captured[:steps]
    if any(x.targets_deg is None for x in used):
        raise ValueError('Teacher pose labels required for offline comparisons')
    labels_deg=np.stack([x.targets_deg for x in used]).astype(float)
    poses=[str(x.pose) for x in used]
    mapper=LatentSynergyMapper(RobotHandModel(), ROOT/'configs/latent_chm6_adaptive.json')
    joint_names=mapper.robot.joint_order
    active_indices=[mapper.teacher.index[x] for x in mapper.teacher.active_order]
    def read(group:str, mode:str):
        folder=(root/'modules'/('adaptive' if group=='B' else 'reachability' if group=='C' else 'credible')
                if group!='E' else root/'spectrum')
        rows=read_run_rows(folder/(mode+'.csv'),steps)
        return rows, get_error_and_timing(rows, joint_names=joint_names,
                                          active_indices=active_indices, labels_deg=labels_deg)
    summary={
        'schema':'mapping_core_evidence_v1',
        'data_path':str(samples_file), 'data_sha256':hash_file(samples_file),
        'frames':steps, 'pose_clusters':len(set(poses)), 'bootstrap_seed':seed,
        'primary_metric':'offline absolute error against preset software-teacher six-DoF posture labels (degrees)',
        'warnings':[
            'No direct measured robot ground truth, independent participants, or physical grasp success.',
            'Historical file reused in previous research; no new blind held-out validation.',
            'Bootstrap over only pose classes is exploratory, NOT evidence of population significance.',
            'Method control spaces, task constraints and optimizer workloads differ; runtimes are descriptive.',
            'Negative and null ablations are preserved rather than excluded.'
        ], 'paired':[], 'stageb':None, 'missing_physics_stagec':('stagec' in groups)}
    for label, group, candidate,baseline in COMPARISONS['modules'] if 'modules' in groups else []:
        a, (err_a,t_a,pipe_a)=read(group,candidate)
        b, (err_b,t_b,pipe_b)=read(group,baseline)
        if [r['frame'] for r in a] != [r['frame'] for r in b]:
            raise ValueError('Run frames do not match')
        stat=pose_paired_bootstrap(err_a-err_b,poses,seed=seed, repeats=bootstrap)
        summary['paired'].append({'comparison':label,'candidate':candidate,'baseline':baseline,
                                  'candidate_teacher_mae_deg':float(np.mean(err_a)),
                                  'baseline_teacher_mae_deg':float(np.mean(err_b)),
                                  'candidate_optimizer_median_ms':float(np.median(t_a)),
                                  'baseline_optimizer_median_ms':float(np.median(t_b)),
                                  'candidate_pipeline_p95_ms':float(np.percentile(pipe_a,95)),
                                  'baseline_pipeline_p95_ms':float(np.percentile(pipe_b,95)),**stat})
    if 'spectrum' in groups:
        for label, group, candidate, baseline in COMPARISONS['spectrum']:
            a,(err_a,t_a,pipe_a)=read(group,candidate)
            b,(err_b,t_b,pipe_b)=read(group,baseline)
            stat=pose_paired_bootstrap(err_a-err_b,poses,seed=seed,repeats=bootstrap)
            spectrum_k=(np.array([float(row['spectrum_selected_k']) for row in a])
                        if 'spectrum_selected_k' in a[0] else np.asarray([], dtype=float))
            accepted_k=(np.array([float(row['spectrum_accepted_k']) for row in a])
                        if 'spectrum_accepted_k' in a[0] else np.asarray([], dtype=float))
            summary['paired'].append({'comparison':label,'candidate':candidate,'baseline':baseline,
                           'candidate_teacher_mae_deg':float(np.mean(err_a)),
                           'baseline_teacher_mae_deg':float(np.mean(err_b)),
                           'candidate_optimizer_median_ms':float(np.median(t_a)),
                           'baseline_optimizer_median_ms':float(np.median(t_b)),
                           'candidate_pipeline_p95_ms':float(np.percentile(pipe_a,95)),
                           'baseline_pipeline_p95_ms':float(np.percentile(pipe_b,95)),
                           'E_dynamic_requested_k_lt6_fraction':(float(np.mean(spectrum_k<6)) if len(spectrum_k) else None),
                           'E_dynamic_accepted_fallback_rate':(float(np.mean(accepted_k==0)) if len(accepted_k) else None),
                           **stat})
    if 'stageb' in groups:
        json_file=root/'stageb/summary.json'
        result=json.loads(json_file.read_text(encoding='utf-8'))
        if result.get('schema')!='stageB_offline_ablation_v1':
            raise ValueError('Unexpected Stage B result schema')
        if result['n_external_records'] != len(captured):
            raise ValueError('Stage B external dataset differs in number of records')
        summary['stageb']={'n_records':result['n_external_records'],
                            'n_poses':result['n_unique_poses'],
                            'summary':result['summary'],
                            'bootstrap':result['pose_cluster_bootstrap'],
                            'warnings':result['warning']}
    root.mkdir(parents=True,exist_ok=True)
    (root/'evidence.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (root/'paired_comparisons.csv').open('w',encoding='utf-8-sig',newline='') as f:
        columns=['comparison','candidate','baseline','candidate_teacher_mae_deg','baseline_teacher_mae_deg',
                 'mean_paired_delta_deg','pose_cluster_count','n_paired_frames','ci95_pose_cluster_bootstrap',
                 'candidate_optimizer_median_ms','baseline_optimizer_median_ms',
                 'candidate_pipeline_p95_ms','baseline_pipeline_p95_ms','E_dynamic_requested_k_lt6_fraction','E_dynamic_accepted_fallback_rate']
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader()
        for row in summary['paired']:
            data={k:row.get(k,'') for k in columns}
            if isinstance(data['ci95_pose_cluster_bootstrap'],list):
                data['ci95_pose_cluster_bootstrap']=';'.join(map(str,data['ci95_pose_cluster_bootstrap']))
            writer.writerow(data)
    lines=['# 核心映射对比：统一数据来源与配对证据报告','',
           f'- 预设姿态标签帧数：**{steps}**，姿态组数：**{len(set(poses))}**。',
           f'- 数据 SHA256：`{summary["data_sha256"]}`。',
           '- **MAE 是相对于软件教师标签的离线误差，不是真实关节真值，也不表示抓握成功。**',
           '- 所有配对差值定义为 **候选方法 − 基线方法**，负数表示离线标签误差更低。',
           '- 区间采用少量姿态组重采样，仅用于探索性敏感度分析，**不是显著性证明**。',
           '- E 与原直接 11 DoF 基线使用不同的耦合/优化空间，速度对比存在结构差异。',
           '', '## 核心配对（新运行结果）','',
           '| 对比（候选 − 基线） | 候选 MAE° | 基线 MAE° | 平均姿态配对差° | 姿态 bootstrap 95% 区间° | 优化器中位耗时 ms（候选 / 基线） |',
           '|---|---:|---:|---:|---|---|']
    for r in summary['paired']:
        interval=r['ci95_pose_cluster_bootstrap']
        cint='不可估计' if interval is None else f'[{interval[0]:+.3f}, {interval[1]:+.3f}]'
        lines.append(f"| {r['comparison']} | {r['candidate_teacher_mae_deg']:.3f} | {r['baseline_teacher_mae_deg']:.3f} | {r['mean_paired_delta_deg']:+.3f} | {cint} | {r['candidate_optimizer_median_ms']:.2f} / {r['baseline_optimizer_median_ms']:.2f} |")
    if summary['stageb']:
        lines+=['', '## Stage B 同一故障掩码对照（新运行结果）','',
               '| 条件 | 原 Ridge6 | 增强 Ridge6 | 原 PCA4 | 增强 PCA4 | 标量融合 | 原 TQHG-v2 |',
               '|---|---:|---:|---:|---:|---:|---:|']
        for row in summary['stageb']['summary']:
            lines.append(f"| {row['condition']} | {row['ridge6_original']:.4f} | {row['ridge6_robust']:.4f} | {row['pca4_original']:.4f} | {row['pca4_robust']:.4f} | {row['calibrated_scalar_blend']:.4f} | {row['tqhg_v2_original']:.4f} |")
        lines +=['','- Stage B 的误差是六维归一化标签 MAE，与上表角度 MAE **不在同一尺度**。',
                 '- 增强与历史非增强模型训练条件不完全相同，不能把差异归于单一算法。',
                 '- 测试仍是之前用过的 24 帧/6 姿态，掩码重复不构成独立受试者。']
    lines+=['','## 下一轮最需要补足的证据','',
            '1. 独立采集连续新数据；训练/调参/测试按受试人、日期或连续序列隔离。',
            '2. 以相同时间预算/约束定义比较算法，报告实际约束违反、风险/守卫触发率和计算时延。',
            '3. 对 D 使用“去不确定性 / 去守卫”两项消融；效果为零也保留负结果。',
            '4. 对 E 同时比较动态选维与真正固定 k=6，分别报告 requested_k、accepted_k、output_rank 和 fallback；旧 spectrum_modes 是输出秩，不代表选维数量。',
            '5. 在 MuJoCo 相同输入与故障条件下运行 Stage C，并公开完整每帧 CSV 和配对汇总；当前尚待完成。','']
    (root/'evidence_report.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary
