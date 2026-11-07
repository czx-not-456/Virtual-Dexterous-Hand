"""Paired Stage C MuJoCo benchmark (requires Windows-side mujoco install).

Each method/condition uses the SAME replay and the SAME explicit invalid
sensor mask. Avoid claiming repeated deterministic replays are independent
human trials. This script never updates models or calibration.
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
METHODS=('robust6','robust4','stagec-alpha','stagec-beta','stagec-dual')
CONDITIONS={'clean':'','mask_one':'1','mask_three':'1,4,7','mask_five':'1,4,7,10,13'}


def _float(x):
    try:return float(x)
    except (TypeError,ValueError):return float('nan')


def summarize(outdir, names, fault_start=120, fault_stop=240):
    results=[]
    for name,condition,method in names:
        json_file=outdir/(name+'_summary.json')
        csv_file=outdir/(name+'.csv')
        if not json_file.exists() or not csv_file.exists():
            raise FileNotFoundError('Missing Stage C output for '+name)
        summ=json.loads(json_file.read_text(encoding='utf-8'))
        with csv_file.open('r',encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        holding=[r for r in rows if 120<=int(r['frame'])<=270 and r.get('task_active','').lower() in ('1','true')]
        fault_rows=[r for r in rows if fault_start<=int(r['frame'])<fault_stop and r.get('task_active','').lower() in ('1','true')]
        recovery_rows=[r for r in rows if fault_stop<=int(r['frame'])<min(fault_stop+60,len(rows)) and r.get('task_active','').lower() in ('1','true')]
        def mean(key, arr=holding, scale=1.):
            vals=[(1. if str(r.get(key)).lower() in ('true','1') else 0. if str(r.get(key)).lower() in ('false','0') else _float(r.get(key))) for r in arr]
            vals=[v for v in vals if v==v]
            return sum(vals)/len(vals)*scale if vals else float('nan')
        results.append({
            'condition':condition,'method':method,'run_name':name,
            'contact_pct':100*_float(summ['contact_frame_ratio']),
            'stable_pct':100*_float(summ['stable_frame_ratio']),
            'stable_success':bool(summ['stable_success_proxy']),
            'stable_streak_frames':int(summ['max_stable_success_streak_frames']),
            'thumb_mean_error_mm_120_270':mean('servo_thumb_error_m',scale=1000),
            'index_mean_error_mm_120_270':mean('servo_index_error_m',scale=1000),
            'w_task_mean_120_270':mean('mapping_w_task'),
            'w_null_mean_120_270':mean('mapping_w_null'),
            'task_rank_mean_120_270':mean('mapping_visible_rank'),
            'projection_leakage_mean_120_270':mean('mapping_stagec_projection_leakage'),
            'fault_window_contact_pct':mean('task_contact_proxy',arr=fault_rows,scale=100.),
            'fault_window_thumb_error_mm':mean('servo_thumb_error_m',arr=fault_rows,scale=1000.),
            'fault_window_stable_pct':mean('hold_stable_proxy',arr=fault_rows,scale=100.),
            'post_recovery_contact_pct':mean('task_contact_proxy',arr=recovery_rows,scale=100.),
            'post_recovery_thumb_error_mm':mean('servo_thumb_error_m',arr=recovery_rows,scale=1000.),
            'p95_latency_ms':_float(summ['p95_pipeline_latency_ms']),
            'fault_channels_1based':';'.join(map(str,summ.get('stagec_fault_channels_1based',[]))),
        })
    out=outdir/'stageC_summary.csv'
    with out.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(results[0]))
        writer.writeheader();writer.writerows(results)
    print('[DONE]',out)
    for r in results:
        print(f"{r['condition']:10} {r['method']:13} contact={r['contact_pct']:6.2f}% stable={r['stable_pct']:6.2f}% "
              f"T={r['thumb_mean_error_mm_120_270']:.3f}mm I={r['index_mean_error_mm_120_270']:.3f}mm "
              f"alpha={r['w_task_mean_120_270']:.4f} beta={r['w_null_mean_120_270']:.4f}")
    print('[CAUTION] One fixed glove replay per condition: descriptive diagnosis only; NOT independent grasp trials.')
    return out


def build_command(executable,glove,outdir,condition,method,steps,start,stop,task="pinch"):
    return [str(executable),'-m','src.main',
       '--input','glove','--glove-samples',str(glove),
       '--sim','mujoco','--task',task,'--steps',str(steps),
       '--output-dir',str(outdir),'--run-name',f'stageC_{task}_{condition}_{method}',
       '--mapping-method',method,
       '--stagec-robust6-model','configs/stageB_robust_ridge6.json',
       '--stagec-gate-model','configs/stageC_task_gate.json',
       '--tqhg-v2-calibration','configs/tqhg_online_v2.json',
       '--coupled-servo-ablation','hold','--servo-trace',
       '--stagec-fault-channels',CONDITIONS[condition],
       '--stagec-fault-start',str(start),'--stagec-fault-stop',str(stop)]


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--glove',type=Path,default=Path('数据手套数据/CHM6_pinch_hold_diagnostic_only.xlsx'))
    ap.add_argument('--conditions',default='clean,mask_three',help='Comma-separated clean,mask_one,mask_three,mask_five')
    ap.add_argument('--methods',default=','.join(METHODS))
    ap.add_argument('--outdir',type=Path,default=Path(__file__).resolve().parent/'results/generated/stageC_mujoco')
    ap.add_argument('--steps',type=int,default=360)
    ap.add_argument('--task',choices=('pinch','wrap','sphere'),default='pinch')
    ap.add_argument('--fault-start',type=int,default=120)
    ap.add_argument('--fault-stop',type=int,default=240)
    ap.add_argument('--dry-run',action='store_true',help='Show commands without requiring MuJoCo')
    a=ap.parse_args(argv)
    conditions=[s.strip() for s in a.conditions.split(',')]
    methods=[s.strip() for s in a.methods.split(',')]
    if any(x not in CONDITIONS for x in conditions) or any(x not in METHODS for x in methods):
        ap.error('unknown condition/method')
    if not a.dry_run and not (ROOT/a.glove).exists() and not a.glove.exists():
        ap.error('Glove diagnostic workbook not found; copy your original 360-frame local file')
    outdir= a.outdir if a.outdir.is_absolute() else ROOT/a.outdir
    names=[]
    for condition in conditions:
        for method in methods:
            name=f'stageC_{a.task}_{condition}_{method}'
            cmd=build_command(sys.executable,a.glove,outdir,condition,method,a.steps,a.fault_start,a.fault_stop,a.task)
            names.append((name,condition,method))
            print('[RUN]',' '.join('"'+x+'"' if ' ' in x else x for x in cmd),flush=True)
            if not a.dry_run:
                subprocess.run(cmd,check=True,cwd=ROOT)
    if not a.dry_run:
        summarize(outdir,names,a.fault_start,a.fault_stop)

if __name__=='__main__':
    main()
