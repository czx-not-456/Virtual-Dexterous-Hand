"""Paired replay of Scheme E and original baselines; teacher-label only.

WARNING: the 24 rows contain six preset posture labels, NOT measured robot
joint tracking accuracy, contact precision, generalization or task success.
"""
import argparse, csv, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.glove.captured_samples import load_captured_samples
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.latent_synergy import LatentSynergyMapper

def run(steps=24,output=None,*,samples=None,source="sensor",modes=None):
    sample=Path(samples or ROOT/'数据手套数据/captured_samples(1).xlsx').resolve()
    captures=load_captured_samples(sample)
    if steps<1 or steps>len(captures): raise ValueError('invalid frames')
    out=Path(output or Path(__file__).resolve().parent/'results/generated/spectrum_benchmark').resolve();out.mkdir(parents=True,exist_ok=True)
    mapper=LatentSynergyMapper(RobotHandModel(), ROOT/'configs/latent_chm6_adaptive.json')
    runs={
        'original_direct':None,  # original 11-DoF optimization, no fixed coupling
        'fixed_PCA4':[],
        'scheme_C':['--reachability-synergy','--reachability-config','configs/reachability_synergy.yaml'],
        'scheme_D':['--credible-contact-synergy','--credible-config','configs/credible_contact_synergy.yaml'],
        'scheme_E':['--task-effect-spectrum','--spectrum-config','configs/task_effect_spectrum.yaml'],
        'E_fixed6':['--task-effect-spectrum','--spectrum-config','configs/task_effect_spectrum_fixed6.yaml'],
        'E_with_lowpass':['--task-effect-spectrum','--spectrum-config','configs/task_effect_spectrum_lowpass.yaml'],
    }
    selected = list(runs) if modes is None else list(modes)
    if not selected or len(selected) != len(set(selected)) or any(m not in runs for m in selected):
        raise ValueError(f'Invalid --modes; available: {list(runs)}')
    results={"scope":"preset-pose software teacher labels; no real tracking/contact ground truth",
             "frame_count":steps, "pose_groups":len(set(x.pose for x in captures[:steps])),
             "sample_path":str(sample),
             "sample_sha256":hashlib.sha256(sample.read_bytes()).hexdigest(),
             "latent_source":source, "modes":selected, "runs":{}}
    for name in selected:
        extra = runs[name]
        cmd=[sys.executable,'-m','src.main','--input','glove','--sim','none',
             '--steps',str(steps),'--glove-samples',str(sample),
             '--output-dir',str(out),'--run-name',name]
        if extra is not None:
            cmd+=['--latent-model','configs/latent_chm6_adaptive.json',
                  '--latent-source',source]+extra
        subprocess.run(cmd,cwd=ROOT,check=True,capture_output=True,text=True)
        with (out/(name+'.csv')).open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        q=np.array([[float(row['q_'+j]) for j in mapper.robot.joint_order] for row in rows])
        targets=np.stack([r.targets_deg for r in captures[:steps]])
        index=[mapper.teacher.index[j] for j in mapper.teacher.active_order]
        mae=float(np.mean(np.abs(np.rad2deg(q[:,index])-targets)))
        times=np.array([float(r['optimizer_ms']) for r in rows])
        pipe=np.array([float(r['pipeline_latency_ms']) for r in rows])
        record={
            'control_label_MAE_deg':mae,'optimizer_median_ms':float(np.median(times)),
            'optimizer_p95_ms':float(np.percentile(times,95)),
            'pipeline_p95_ms':float(np.percentile(pipe,95)),
            'coupling_error_max_rad':max(float(np.max(np.abs(q[:,mapper.teacher.index[child]]-
                  gain*q[:,mapper.teacher.index[driver]])))
                  for driver,child,gain in mapper.teacher.couplings),
            'limit_violations':int(np.sum((q<mapper.robot.ranges[:,0]-1e-8)|(q>mapper.robot.ranges[:,1]+1e-8))),
        }
        if name.startswith('scheme_E') or name.startswith('E_'):
            record.update({
                'selected_modes_mean':float(np.mean([float(r['spectrum_selected_k']) for r in rows])) if 'spectrum_selected_k' in rows[0] else None,
                'accepted_modes_mean':float(np.mean([float(r['spectrum_accepted_k']) for r in rows])) if 'spectrum_accepted_k' in rows[0] else None,
                'output_rank_mean':float(np.mean([float(r['spectrum_output_rank']) for r in rows])) if 'spectrum_output_rank' in rows[0] else None,
                'low_rank_requested_fraction':float(np.mean([float(r['spectrum_selected_k'])<6 for r in rows])) if 'spectrum_selected_k' in rows[0] else None,
                'fallback_rate':float(np.mean([float(r['spectrum_fallback']) for r in rows])) if 'spectrum_fallback' in rows[0] else None,
                'selected_modes_histogram':{str(i):sum(float(r['spectrum_selected_k'])==i for r in rows) for i in range(1,7)} if 'spectrum_selected_k' in rows[0] else {},
                'final_guard_rate':float(np.mean([float(r['spectrum_post_guard']) for r in rows])),
                'internal_fallback_rate':float(np.mean([float(r['reach_fallback_used']) for r in rows])),
                'final_task_guard_pass_rate':float(np.mean([float(r['spectrum_post_tip_m']) <= float(r['spectrum_post_baseline_m']) + .0001001 for r in rows])),
                'speed_box_pass_rate':float(np.mean([float(r['spectrum_speed_valid']) for r in rows])),
                'local_spectral_residual_mean_m':float(np.mean([float(r['spectrum_omission_m']) for r in rows])),
            })
        results['runs'][name]=record
        print(name,record,flush=True)
    output_path=out/'schemeE_paired_comparison.json'
    output_path.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    return output_path

if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples', type=Path, default=ROOT/'数据手套数据/captured_samples(1).xlsx')
    ap.add_argument('--steps', type=int, default=24)
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'results/generated/spectrum_benchmark')
    ap.add_argument('--source', choices=('sensor','control','blend'), default='sensor')
    ap.add_argument('--modes', default='', help='Comma-separated subset; empty means legacy full set')
    args = ap.parse_args()
    print(run(steps=args.steps,output=args.out,samples=args.samples,source=args.source,
              modes=args.modes.split(',') if args.modes else None))
