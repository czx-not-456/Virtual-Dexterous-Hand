"""MuJoCo task validation for Scheme E using one captured glove recording.

Primary fair comparison: E_dynamic vs E_fixed6 (same six-control mapper,
solver, guards, task and captured input; only dimension rule differs).
Other modes may be added as contextual baselines. Metrics are simulation
proxies, not hardware performance or independent-user ground truth.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.实验08_任务效应谱方案对比.benchmark_scheme_e_research import CONFIGS,sha256


def command(mode,task,samples,out,steps,args):
    cmd=[sys.executable,'-m','src.main','--input','glove','--glove-samples',str(samples),
         '--sim','mujoco','--task',task,'--steps',str(steps),'--latent-source','sensor',
         '--output-dir',str(out),'--run-name',f'{task}_{mode}']
    config=CONFIGS[mode]
    if mode=='original_direct':
        pass
    elif mode=='fixed_PCA4':
        cmd += ['--latent-model','configs/latent_chm6_adaptive.json']
    elif config:
        cmd += ['--latent-model','configs/latent_chm6_adaptive.json','--task-effect-spectrum',
                '--spectrum-config',config]
    if args.no_auto_reset: cmd.append('--no-auto-reset-object')
    if args.no_inactive_reset: cmd.append('--no-inactive-object-reset')
    if args.render: cmd.append('--render')
    if args.realtime: cmd.append('--realtime')
    return cmd


def read_csv(p):
    with p.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples',type=Path,default=ROOT/'数据手套数据/records/chm6_calibrations/20260904_113022_G15RB010901_right/captured_samples.csv')
    ap.add_argument('--tasks',default='pinch,wrap')
    ap.add_argument('--modes',default='E_dynamic,E_fixed6')
    ap.add_argument('--steps',type=int,default=0,help='0 uses all rows in the captured recording')
    ap.add_argument('--out',type=Path,default=ROOT/'outputs/scheme_e_mujoco')
    ap.add_argument('--deadline-ms',type=float,default=1000/60)
    ap.add_argument('--no-auto-reset',action='store_true',default=True)
    ap.add_argument('--no-inactive-reset',action='store_true',default=True)
    ap.add_argument('--render',action='store_true')
    ap.add_argument('--realtime',action='store_true')
    ap.add_argument('--process-timeout',type=float,default=240.0)
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args(argv)
    if not args.samples.is_file(): ap.error(f'missing sample file: {args.samples}')
    tasks=args.tasks.split(',');modes=args.modes.split(',')
    if any(t not in {'pinch','wrap'} for t in tasks):ap.error('tasks must be pinch,wrap')
    if len(set(tasks))!=len(tasks) or len(set(modes))!=len(modes):ap.error('duplicate tasks or modes')
    if any(m not in CONFIGS for m in modes):ap.error(f'modes must be from {list(CONFIGS)}')
    with args.samples.open('r',encoding='utf-8-sig',newline='') as f:
        steps=sum(1 for _ in csv.DictReader(f))
    steps=args.steps or steps
    if steps<1:ap.error('empty recording')
    plan=[]
    for task in tasks:
        for mode in modes:
            dest=args.out/task/mode
            cmd=command(mode,task,args.samples,dest,steps,args)
            plan.append((task,mode,dest,cmd))
    print('[INPUT]',args.samples,'frames=',steps,'sha256=',sha256(args.samples))
    print('[NOTE] Simulation-only metrics. No hardware-grasp or independent-subject claim.')
    for task,mode,dest,cmd in plan:
        print('[PLAN]',task,mode,subprocess.list2cmdline(cmd) if os.name=='nt' else ' '.join(cmd))
    if args.dry_run:
        print('[DRY RUN] MuJoCo tasks were not executed');return 0
    args.out.mkdir(parents=True,exist_ok=True)
    manifest={'schema':'scheme_e_mujoco_validation_v1','sample_path':str(args.samples.resolve()),
              'sample_sha256':sha256(args.samples),'steps':steps,'tasks':tasks,'modes':modes,
              'deadline_ms':args.deadline_ms,'auto_reset_disabled':args.no_auto_reset,
              'inactive_reset_disabled':args.no_inactive_reset,'runs':[],'complete':False,
              'scope':'MuJoCo simulated task performance; fixed-base/contact proxies are not hardware evidence'}
    (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    summary_rows=[]
    for task,mode,dest,cmd in plan:
        dest.mkdir(parents=True,exist_ok=True)
        log=dest/'run.log';start=time.perf_counter()
        env=os.environ.copy();env['PYTHONUTF8']='1';env['PYTHONIOENCODING']='utf-8'
        print('[RUN]',task,mode,flush=True)
        try:
            with log.open('w',encoding='utf-8',errors='replace') as f:
                proc=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env,
                                    timeout=args.process_timeout,check=False)
        except subprocess.TimeoutExpired:
            manifest['runs'].append({'task':task,'mode':mode,'exit_code':'timeout','log':str(log)})
            (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
            raise RuntimeError(f'{task}/{mode} timed out; see {log}')
        wall=time.perf_counter()-start
        record={'task':task,'mode':mode,'argv':cmd,'exit_code':proc.returncode,'wall_s':wall,'log':str(log)}
        manifest['runs'].append(record)
        (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        if proc.returncode!=0:raise RuntimeError(f'{task}/{mode} failed; see {log}')
        csv_path=dest/f'{task}_{mode}.csv';summary_path=dest/f'{task}_{mode}_summary.json'
        if not csv_path.is_file() or not summary_path.is_file():raise FileNotFoundError(f'missing outputs under {dest}')
        rows=read_csv(csv_path);sj=json.loads(summary_path.read_text(encoding='utf-8'))
        pipeline=np.asarray([float(r.get('pipeline_latency_ms','nan')) for r in rows])
        pipeline=pipeline[np.isfinite(pipeline)]
        def finite_col(key):
            x=np.asarray([float(r.get(key,'nan')) for r in rows]);return x[np.isfinite(x)]
        rank=finite_col('spectrum_output_rank');fallback=finite_col('spectrum_fallback')
        reach_err=finite_col('reach_task_linear_error_m')
        servo_err=finite_col('servo_mean_error_m')
        unreachable_err=finite_col('reach_unreachable_error_m')
        summary_rows.append({'task':task,'policy':mode,'frames':len(rows),'wall_s':wall,
          'contact_success_proxy':sj.get('contact_success_proxy'),'stable_success_proxy':sj.get('stable_success_proxy'),
          'contact_frame_ratio':sj.get('contact_frame_ratio'),'stable_frame_ratio':sj.get('stable_frame_ratio'),
          'mean_task_linear_error_mm':float(np.mean(reach_err)*1000) if len(reach_err) else None,
          'p95_task_linear_error_mm':float(np.percentile(reach_err,95)*1000) if len(reach_err) else None,
          'mean_servo_error_mm':float(np.mean(servo_err)*1000) if len(servo_err) else None,
          'p95_servo_error_mm':float(np.percentile(servo_err,95)*1000) if len(servo_err) else None,
          'mean_unreachable_error_mm':float(np.mean(unreachable_err)*1000) if len(unreachable_err) else None,
          'object_displacement_m_max':float(np.max(finite_col('object_displacement_m'))) if len(finite_col('object_displacement_m')) else None,
          'pipeline_p95_ms':float(np.percentile(pipeline,95)) if len(pipeline) else None,
          'pipeline_p99_ms':float(np.percentile(pipeline,99)) if len(pipeline) else None,
          'deadline_miss_rate':float(np.mean(pipeline>args.deadline_ms)) if len(pipeline) else None,
          'actual_output_low_rank_fraction':float(np.mean(rank<6)) if len(rank) else None,
          'fallback_rate':float(np.mean(fallback)) if len(fallback) else None,
          'csv':str(csv_path),'summary':str(summary_path)})
        print('[OK]',task,mode,'contact=',sj.get('contact_frame_ratio'),'stable=',sj.get('stable_success_proxy'),flush=True)
    with (args.out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary_rows[0]));w.writeheader();w.writerows(summary_rows)
    lines=['# Scheme E MuJoCo task validation','',f'- Sample recording: `{args.samples.name}`; SHA256 `{sha256(args.samples)}`.',
           f'- Tasks: {", ".join(tasks)}; modes: {", ".join(modes)}; frames/run: {steps}.',
           f'- 60Hz deadline: {args.deadline_ms:.3f} ms. Deadline miss is a latency statistic, not a hard real-time certification.',
           '- MuJoCo simulated task metrics are not hardware-grasp evidence. The success proxy in this project is fixed-base/contact based, not lift/transport.',
           '', '| Task | Method | Contact ratio | Stable proxy | Task error mean/P95 mm | Servo error mean/P95 mm | Pipeline P95 ms | 60Hz miss rate | Actual low-rank fraction | Fallback rate |','|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in summary_rows:
        lines.append(f"| {r['task']} | {r['policy']} | {r['contact_frame_ratio']} | {r['stable_success_proxy']} | {r['mean_task_linear_error_mm']}/{r['p95_task_linear_error_mm']} | {r['mean_servo_error_mm']}/{r['p95_servo_error_mm']} | {r['pipeline_p95_ms']} | {r['deadline_miss_rate']} | {r['actual_output_low_rank_fraction']} | {r['fallback_rate']} |")
    (args.out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest['complete']=True;manifest['summary']='summary.csv';manifest['report']='report.md'
    (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[DONE]',args.out/'report.md');return 0

if __name__=='__main__': raise SystemExit(main())
