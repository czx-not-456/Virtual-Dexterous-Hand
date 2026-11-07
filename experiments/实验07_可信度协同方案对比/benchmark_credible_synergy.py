"""Reproducible *label-only* offline tests C vs D and D ablations.

ALL 24 frames are six preset poses; no independent human kinematics or real
contact measurement. Subprocess runs use identical sampling and rate limiting.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.glove.captured_samples import load_captured_samples
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.latent_synergy import LatentSynergyMapper


def metrics(rows, refs, model):
    q=np.array([[float(row['q_'+name]) for name in model.robot.joint_order]
                for row in rows])
    targets=np.stack([r.targets_deg for r in refs[:len(rows)]])
    indices=[model.teacher.index[name] for name in model.teacher.active_order]
    u_deg=np.rad2deg(q[:,indices]); err=np.abs(u_deg-targets)
    coupled=max(float(np.max(np.abs(q[:,model.teacher.index[child]]-
                 gain*q[:,model.teacher.index[driver]])))
                for driver,child,gain in model.teacher.couplings)
    stats={
      'frames':len(rows), 'teacher_label_MAE_deg':float(err.mean()),
      'pose_label_MAE_deg':{pose:float(err[[i for i,r in enumerate(refs[:len(rows)])
                                            if r.pose==pose]].mean())
                             for pose in sorted(set(r.pose for r in refs[:len(rows)]))},
      'joint_limit_violations':int(((q<model.robot.ranges[:,0]-1e-8)|
                                  (q>model.robot.ranges[:,1]+1e-8)).sum()),
      'max_coupling_deviation_rad':coupled,
      'optimizer_median_ms':float(np.median([float(r['optimizer_ms']) for r in rows])),
      'pipeline_p95_ms':float(np.percentile([float(r['pipeline_latency_ms'])
                                                for r in rows],95)),
    }
    if np.isfinite(float(rows[0].get('reach_variables','nan'))):
      stats['NLP_variables_mean']=float(np.mean([float(r['reach_variables']) for r in rows]))
      stats['output_rank_mean']=float(np.mean([float(r['reach_command_rank']) for r in rows]))
      stats['sparse_residual_rate']=float(np.mean([float(r['reach_residual_axes'])>0 for r in rows]))
    if np.isfinite(float(rows[0].get('trust_risk_bound_m','nan'))):
      stats['conditional_linear_risk_bound_mean_m_equivalent']=float(np.mean([
                            float(r['trust_risk_bound_m']) for r in rows]))
      stats['conditional_budget_pass_rate']=float(np.mean([
                            float(r['trust_feasible']) for r in rows]))
      stats['nonlinear_guard_rate']=float(np.mean([
                            float(r['trust_guard']) for r in rows]))
      stats['contact_phase_unobserved_fraction']=float(np.mean([
                            float(r['trust_contact_phase']) == -1 for r in rows]))
    return stats


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=24)
    parser.add_argument('--source',choices=['sensor','control','blend'],default='sensor')
    parser.add_argument('--samples',type=Path,default=ROOT/'数据手套数据/captured_samples(1).xlsx')
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results/generated/credible_comparison')
    args=parser.parse_args();args.samples=args.samples.resolve();args.out=args.out.resolve()
    captures=load_captured_samples(args.samples)
    if args.steps<1 or args.steps>len(captures):
        raise ValueError('no looping over the same dataset in benchmark')
    if any(x.targets_deg is None for x in captures[:args.steps]):
        raise ValueError('teacher labels required')
    args.out.mkdir(parents=True,exist_ok=True)
    mapper=LatentSynergyMapper(RobotHandModel(),ROOT/'configs/latent_chm6_adaptive.json')
    modes={
      'fixed_PCA_k4':[],
      'reachability_C':['--reachability-synergy','--reachability-config',
                        'configs/reachability_synergy.yaml'],
      'credible_D_diagnostic':['--credible-contact-synergy','--credible-config',
                   'configs/credible_contact_synergy.yaml'],
      'credible_D_risk_enforced_experimental':['--credible-contact-synergy','--credible-config',
                   'configs/credible_risk_enforced_experimental.yaml'],
      'D_without_uncertainty':['--credible-contact-synergy','--credible-config',
                   'configs/credible_no_uncertainty.yaml'],
      'D_without_nonlinear_guard':['--credible-contact-synergy','--credible-config',
                   'configs/credible_no_guard.yaml'],
    }
    results={'task':'Offline six-label control accuracy, NOT robot tracking or grasp success',
             'input':str(args.samples),'source':args.source,'contact_available':False,
             'runs':{}}
    for name,extra in modes.items():
        cmd=[sys.executable,'-m','src.main','--input','glove','--sim','none',
             '--steps',str(args.steps),'--latent-model','configs/latent_chm6_adaptive.json',
             '--glove-samples',str(args.samples),'--latent-source',args.source,
             '--output-dir',str(args.out),'--run-name',name]+extra
        subprocess.run(cmd,cwd=ROOT,check=True,capture_output=True,text=True)
        with (args.out/(name+'.csv')).open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        results['runs'][name]=metrics(rows,captures,mapper)
        print(name, results['runs'][name]['teacher_label_MAE_deg'],
              results['runs'][name]['optimizer_median_ms'],flush=True)
    path=args.out/'schemeD_comparison.json'
    path.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Saved',path)

if __name__=='__main__':main()
