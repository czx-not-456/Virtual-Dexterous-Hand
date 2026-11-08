"""Stage B held-out offline ablation: common masks, per-pose paired statistics.

This is *not* a MuJoCo task-space test (simulator=None => P=0, N=I).
24 samples describe only six preset postures. The 20 repeated masks are NOT
480 independent human movement samples. No tuning or retraining occurs here.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.glove.captured_samples import load_captured_samples
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.latent_synergy import CONTROL_LIMITS_DEG
from src.retargeting.tqhg_v2 import TQHGv2OnlineMapper,masked_features
from scripts.stageB_common import CONDITIONS,METHODS,paired_cases,predict_blends


def pose_cluster_ci(records, candidate, comparator, rng, n_boot=3000):
    """Exploratory paired bootstrap resampling the SIX pose groups (not mask rows)."""
    by_pose=defaultdict(list)
    for r in records:
        by_pose[r['pose']].append(float(r[candidate])-float(r[comparator]))
    poses=list(by_pose)
    pose_means=np.array([np.mean(by_pose[p]) for p in poses])
    if len(pose_means)<2:
        return None,None
    picks=rng.integers(0,len(poses),size=(n_boot,len(poses)))
    means=np.mean(pose_means[picks],axis=1)
    return tuple(float(x) for x in np.quantile(means,[0.025,0.975]))


def write_csv(path,rows,columns):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        w.writeheader();w.writerows(rows)


def chart(outdir,per_pose):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[NOTE] matplotlib not available; CSV/JSON outputs still produced')
        return
    methods=['ridge6_original','ridge6_robust','pca4_robust','calibrated_scalar_blend','tqhg_v2_original']
    conds=[c for c,_ in CONDITIONS]
    fig,ax=plt.subplots(figsize=(11.5,5))
    offsets=np.linspace(-.30,.30,len(methods))
    width=.12
    for i,m in enumerate(methods):
        values=[]
        for c in conds:
            z=[float(p[m]) for p in per_pose if p['condition']==c]
            values.append(float(np.mean(z)))
        ax.bar(np.arange(len(conds))+offsets[i],values,width=width,label=m)
    ax.set_xticks(range(len(conds)),conds)
    ax.set_ylabel('Mean normalized 6D teacher-label MAE (lower better)')
    ax.set_title('Stage B held-out six-pose offline ablation')
    ax.legend(fontsize=7,ncol=2)
    fig.tight_layout()
    fig.savefig(outdir/'mae_by_condition.png',dpi=160)
    plt.close(fig)
    # Distribution across six pose-level means, not repeated mask-level samples.
    fig,axs=plt.subplots(1,len(conds),figsize=(14,4.4),sharey=True)
    for ax,c in zip(axs,conds):
        values=[[float(p[m]) for p in per_pose if p['condition']==c]
                for m in ('ridge6_robust','pca4_robust','calibrated_scalar_blend')]
        try:
            ax.boxplot(values,tick_labels=['robust6','robust4','blend'],showmeans=True)
        except TypeError:  # matplotlib < 3.9
            ax.boxplot(values,labels=['robust6','robust4','blend'],showmeans=True)
        ax.set_title(c)
        ax.tick_params(axis='x',labelrotation=20)
    axs[0].set_ylabel('Mean normalized MAE per pose')
    fig.suptitle('Stage B: six-pose error distribution (not 20 independent trials)')
    fig.tight_layout()
    fig.savefig(outdir/'pose_error_distribution.png',dpi=160)
    plt.close(fig)


def evaluate(test_path,model_v2,model_stageB,outdir,*,trials=20,seed=20261025,bootstrap=3000):
    if trials<1 or bootstrap<100:raise ValueError('trials>=1 and bootstrap>=100 required')
    test_path,model_v2,model_stageB,outdir=map(Path,(test_path,model_v2,model_stageB,outdir))
    v2cfg=json.loads(model_v2.read_text(encoding='utf-8'))
    newcfg=json.loads(model_stageB.read_text(encoding='utf-8'))
    if newcfg.get('schema')!='stageB_robust_ridge6_v1':raise ValueError('Train the Stage B robust Ridge6 model first')
    if v2cfg.get('schema')!='tqhg_online_calibration_v2':raise ValueError('Expected TQHG-v2 model')
    if (int(newcfg['provenance']['seed']) != int(v2cfg['provenance']['seed']) or
        int(newcfg['provenance']['train_count']) != 124):
        raise ValueError('Training data/seed mismatch; comparison would not be fairness-controlled')
    if not np.allclose(newcfg['sensor_medians_normalized'],
                       v2cfg['robust_expert']['sensor_medians_normalized']):
        raise ValueError('Imputation statistics differ between robust experts')
    samples=load_captured_samples(test_path)
    if any(s.targets_deg is None for s in samples):raise ValueError('External test labels missing')
    sensors=np.asarray([s.sensors_deg for s in samples],float)
    labels=np.asarray([s.targets_deg for s in samples],float)/CONTROL_LIMITS_DEG
    if not np.isfinite(sensors).all() or not np.isfinite(labels).all():
        raise ValueError('Invalid test sensor or label')
    model=TQHGv2OnlineMapper(RobotHandModel(),model_v2)
    rng=np.random.default_rng(seed)
    outdir.mkdir(parents=True,exist_ok=True)
    details=[]
    for condition,missing,trial,i,mask in paired_cases(sensors,rng,trials):
        # Use the production sensor-quality path, as done in original Stage A.
        model.reset()
        sn,measured,_,_=model.quality.observe(sensors[i],mask)
        features=masked_features(sn,measured,model.medians)
        predictions=predict_blends(features,v2cfg['base_v1'],v2cfg,newcfg,condition)
        model.reset()
        predictions['tqhg_v2_original']=model.predict(sensors[i],method='tqhg-v2',
                      sensor_valid_mask=mask,simulator=None)
        # Original TQHG-v2 output and new conditioned scalar gate both based
        # exclusively on PRE-HELD-OUT CALIBRATION; test labels only score them.
        r={'condition':condition,'missing_count':missing,'trial':trial,'sample_index':int(samples[i].sample_index),
           'pose':samples[i].pose,'missing_channels_1based':';'.join(str(j+1) for j in np.where(~mask)[0]),
           'stageB_weight_calibration':newcfg['calibrated_scalar_weights'][condition],
           'legacy_gate_w_task':float(model.last_gate['w_task']),
           'legacy_gate_w_null':float(model.last_gate['w_null'])}
        for m in METHODS:
            u=np.asarray(predictions[m],float)
            if u.shape != (6,) or not np.isfinite(u).all():
                raise ValueError(f'Nonfinite or wrong dimensional prediction {m}')
            r[m]=float(np.mean(np.abs(u-labels[i])))
        r['best_training_matched_single'] = min(r['ridge6_robust'],r['pca4_robust'])
        r['oracle_diagnostic']=r['best_training_matched_single']
        r['winner_training_matched'] = ('ridge6_robust' if r['ridge6_robust'] < r['pca4_robust']-1e-10
                 else 'pca4_robust' if r['pca4_robust'] < r['ridge6_robust']-1e-10 else 'tie')
        details.append(r)
    columns=list(details[0]);write_csv(outdir/'paired_cases.csv',details,columns)
    per_pose=[];summary=[];stats=[]
    bootstrap_rng=np.random.default_rng(seed+7100)
    for condition,_ in CONDITIONS:
        group=[r for r in details if r['condition']==condition]
        for pose in dict.fromkeys(r['pose'] for r in group):
            subset=[r for r in group if r['pose']==pose]
            per_pose.append({'condition':condition,'pose':pose,'cases':len(subset),
                              **{m:float(np.mean([r[m] for r in subset])) for m in METHODS}})
        item={'condition':condition,'paired_cases':len(group),
              'unique_test_frames':len({r['sample_index'] for r in group}),
              'unique_poses':len({r['pose'] for r in group}),
              'oracle_diagnostic':float(np.mean([r['oracle_diagnostic'] for r in group])),
              'ridge6_robust_win_pct':100.*sum(r['winner_training_matched']=='ridge6_robust' for r in group)/len(group),
              'pca4_robust_win_pct':100.*sum(r['winner_training_matched']=='pca4_robust' for r in group)/len(group),
              'blend_calibration_weight':float(newcfg['calibrated_scalar_weights'][condition])}
        for m in METHODS:item[m]=float(np.mean([r[m] for r in group]))
        summary.append(item)
        for m in METHODS:
            if m=='pca4_robust':continue
            a,b=pose_cluster_ci(group,m,'pca4_robust',bootstrap_rng,bootstrap)
            delta=item[m]-item['pca4_robust']
            stats.append({'condition':condition,'method':m,'comparison':'pca4_robust',
                          'mean_paired_MAE_difference':delta,
                          'pose_cluster_bootstrap_95ci_low':a,'pose_cluster_bootstrap_95ci_high':b,
                          'cluster_unit':'pose','clusters':len({r['pose'] for r in group}),
                          'note':'exploratory six-pose cluster bootstrap; NOT a publication-grade confidence interval'})
    write_csv(outdir/'summary_by_condition.csv',summary,list(summary[0]))
    write_csv(outdir/'summary_by_pose.csv',per_pose,list(per_pose[0]))
    write_csv(outdir/'paired_pose_bootstrap.csv',stats,list(stats[0]))
    chart(outdir,per_pose)
    payload={'schema':'stageB_offline_ablation_v1','test_path':str(test_path),
       'models':{'v2':str(model_v2),'stageB_robust_ridge6':str(model_stageB)},
       'seed':seed,'fault_mask_repeats':trials,'n_external_records':len(samples),
       'n_unique_poses':len(set(x.pose for x in samples)),
       'metric':'MAE vs normalized 6D software-teacher controls, not robot ground truth',
       'task_projection':'simulator=None -> P=0,N=I; cannot infer alpha-vs-beta contribution',
       'calibration':'condition-wise scalar MSE weight fitted ONLY on 66 historical calibration rows',
       'bootstrap':'pose clusters (six); exploratory intervals only; repeated masks are not independent subjects',
       'warning':'Same six-pose held-out file was previously inspected in Stage A; do NOT treat as fresh untouched final validation. Collect new independent gesture sequences.',
       'summary':summary,'pose_cluster_bootstrap':stats}
    (outdir/'summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    report=['# 阶段B 离线消融报告','',
            '> 软件教师6维标签的归一化MAE；24条记录/6种姿态，显式合成缺失。不是机器人抓握性能。','',
            '| 缺失条件 | 原Ridge6 | 增强Ridge6 | 原PCA4 | 增强PCA4 | 固定混合 | 校准标量融合 | 原TQHG-v2 | 理想双专家选择* |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for q in summary:
        names=('ridge6_original','ridge6_robust','pca4_original','pca4_robust','fixed_50_50','calibrated_scalar_blend','tqhg_v2_original','oracle_diagnostic')
        report.append('| '+q['condition']+' | '+' | '.join(f"{q[n]:.6f}" for n in names)+' |')
    report+=['','*理想选择器直接根据测试标签逐样本取两个增强专家的较小误差，不可部署。',
             '', '## 重要说明',
             '- 增强Ridge6和增强PCA4采用完全相同的124条训练数据、相同6次扩增观测（共744训练行）、同一个缺失掩码生成随机种子、同一30维输入与Ridge惩罚系数2.0。',
             '- 原Ridge6、原PCA4没有同等增强，只是历史参照；二者与增强模型的差异不能直接归因于维数。',
             '- 校准标量融合参数只用另外66条历史校准记录，独立测试结果没有参与拟合。该融合没有任务雅可比，不是双权重TQHG。',
             '- 这是使用之前已查看过的24条外部数据的诊断性复测，不应报告为新的盲测泛化能力。',
             '- 20次故障掩码重复共享同24条记录，不能当作480条独立测试运动。',
             '- bootstrap按6种姿态聚类，只有6组；区间只作探索性不确定性提示，不用于正式显著性宣称。',
             '- 任务投影仍是P=0,N=I，本实验不能证明α和β分别改善机器人任务。',
             '', '## 下一阶段建议',
             '采集全新、连续动作与独立批次手套数据，预注册模型和评估指标，再结合MuJoCo实际雅可比测试任务方向与互补方向。']
    (outdir/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(f'INDEPENDENT-BATCH REUSE: {test_path}, records={len(samples)}, poses={payload["n_unique_poses"]}, masks={trials}, seed={seed}')
    print('OFFLINE teacher-label MAE; NO MuJoCo projector. Historical 24 records already inspected in Stage A.')
    print('condition       old_ridge robust6 old_pca  robust4 fixed50 cal_blend old_tqhg oracle*  B_win%')
    for s in summary:
        print(f"{s['condition']:<15} {s['ridge6_original']:.6f} {s['ridge6_robust']:.6f} {s['pca4_original']:.6f} {s['pca4_robust']:.6f} {s['fixed_50_50']:.6f} {s['calibrated_scalar_blend']:.6f} {s['tqhg_v2_original']:.6f} {s['oracle_diagnostic']:.6f} {s['pca4_robust_win_pct']:.1f}")
    print('[DONE]',outdir)
    print('[WARNING] *Oracle sees test labels; repeated masks are NOT independent trials. Six-pose bootstrap is exploratory.')
    return payload

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--test',type=Path,default=ROOT/'数据手套数据/captured_samples(1).xlsx')
    ap.add_argument('--model-v2',type=Path,default=ROOT/'configs/tqhg_online_v2.json')
    ap.add_argument('--robust-ridge6',type=Path,default=ROOT/'configs/stageB_robust_ridge6.json')
    ap.add_argument('--outdir',type=Path,default=Path(__file__).resolve().parent/'results/generated')
    ap.add_argument('--trials',type=int,default=20)
    ap.add_argument('--seed',type=int,default=20261025)
    ap.add_argument('--bootstrap',type=int,default=3000)
    a=ap.parse_args()
    evaluate(a.test,a.model_v2,a.robust_ridge6,a.outdir,trials=a.trials,seed=a.seed,bootstrap=a.bootstrap)
