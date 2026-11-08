"""Core *offline* mapping ablations: Stage B, B/C/D modules, Scheme E.

Runs paired settings with the same input, preserves logs, and writes a
pose-cluster descriptive report. Stage C MuJoCo is deliberately separate.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from experiments.core_mapping_evidence import build_evidence, hash_file

GROUPS=('stageb','modules','spectrum')


def plan_commands(python: str, samples: Path, out: Path, groups:set[str], steps:int,source:str,
                  *,seed:int,bootstrap:int,mask_repeats:int) -> list[tuple[str,list[str]]]:
    scripts={
        'stageb':ROOT/'scripts/evaluate_stageB_ablation.py',
        'B':ROOT/'experiments/benchmark_adaptive_synergy.py',
        'C':ROOT/'experiments/benchmark_reachability_synergy.py',
        'D':ROOT/'experiments/benchmark_credible_synergy.py',
        'E':ROOT/'experiments/benchmark_task_effect_spectrum.py'}
    cmds=[]
    if 'stageb' in groups:
        cmds.append(('StageB', [python,str(scripts['stageb']), '--test',str(samples),
               '--outdir',str(out/'stageb'),'--trials',str(mask_repeats), '--seed',str(seed),
               '--bootstrap',str(bootstrap)]))
    if 'modules' in groups:
        for name,folder in [('B','adaptive'),('C','reachability'),('D','credible')]:
            cmds.append((name,[python,str(scripts[name]), '--samples',str(samples),
                  '--steps',str(steps), '--source',source,'--out',str(out/'modules'/folder)]))
    if 'spectrum' in groups:
        cmds.append(('E',[python,str(scripts['E']), '--samples',str(samples),'--steps',str(steps),
                  '--source',source,'--out',str(out/'spectrum'),
                  '--modes','original_direct,fixed_PCA4,scheme_E,E_fixed6,E_with_lowpass']))
    return cmds


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--groups',default=','.join(GROUPS),help='stageb,modules,spectrum (comma-separated)')
    p.add_argument('--samples',type=Path,default=ROOT/'数据手套数据/captured_samples(1).xlsx')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/mapping_core')
    p.add_argument('--steps',type=int,default=24,help='Offline B/C/D/E records. Stage B always scores full external file.')
    p.add_argument('--source',choices=('sensor','control','blend'),default='sensor')
    p.add_argument('--mask-repeats',type=int,default=20)
    p.add_argument('--seed',type=int,default=20261108)
    p.add_argument('--bootstrap',type=int,default=3000)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--report-only',action='store_true',help='Validate existing CSV files and rebuild report; no simulations')
    args=p.parse_args(argv)
    groups_list=args.groups.split(',')
    if not groups_list or len(set(groups_list))!=len(groups_list) or any(x not in GROUPS for x in groups_list):
        p.error('Select one or more unique groups: '+','.join(GROUPS))
    if args.steps<1 or args.mask_repeats<1 or args.bootstrap<100:
        p.error('steps/mask-repeats must be positive and bootstrap >=100')
    args.samples=args.samples.resolve();args.out=args.out.resolve()
    if not args.samples.is_file():
        p.error(f'Missing samples: {args.samples}')
    cmds=plan_commands(sys.executable,args.samples,args.out,set(groups_list),args.steps,args.source,
                       seed=args.seed,bootstrap=args.bootstrap,mask_repeats=args.mask_repeats)
    # A default smoke / dry-run should not create output files.
    print('[INPUT]',args.samples)
    print('[SHA256]',hash_file(args.samples))
    print('[WARNING] Six posture labels / reused historic data. No physical grasp evidence.')
    for name,cmd in cmds:
        print('[PLAN]',name,subprocess.list2cmdline(cmd) if os.name=='nt' else ' '.join(map(str,cmd)),flush=True)
    if args.dry_run:
        print('[DRY RUN] No benchmarks were executed. Stage C: python scripts/run_stageC_mujoco.py --dry-run')
        return 0
    if args.report_only:
        manifest_file=args.out/'manifest.json'
        if not manifest_file.is_file():
            p.error('report-only requires existing manifest.json; refusing to combine unknown data')
        record=json.loads(manifest_file.read_text(encoding='utf-8'))
        if (record.get('samples_sha256') != hash_file(args.samples) or
            record.get('steps') != args.steps or
            set(record.get('groups', [])) != set(groups_list) or
            record.get('source') != args.source):
            p.error('Existing manifest metadata does not match selected input/groups/source/steps')
        report=build_evidence(args.out,args.samples,set(groups_list),steps=args.steps,
                              seed=args.seed,bootstrap=args.bootstrap)
        record['posthoc_csv_analysis_validated']=True
        record['posthoc_note']='This validates existing CSV content, not missing process exit codes or independence.'
        manifest_file.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        print('[REPORTED] Validated existing CSVs; no new benchmark runs:',args.out/'evidence_report.md')
        return 0
    args.out.mkdir(parents=True,exist_ok=True)
    manifest={
        'schema':'core_mapping_suite_v1',
        'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'source_archive':'v1.3.4 candidate',
        'groups':groups_list,'steps':args.steps,'source':args.source,
        'samples':str(args.samples),'samples_sha256':hash_file(args.samples),
        'python':sys.version,'platform':platform.platform(), 'seed':args.seed,
        'bootstrap_draws':args.bootstrap, 'fault_mask_repeats':args.mask_repeats,
        'commands':[],'complete':False}
    manifest_path=args.out/'manifest.json'
    def save():manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    for name,cmd in cmds:
        print('[RUN]',name,flush=True)
        record={'name':name,'argv':cmd,'exit_code':None,'stdout_log':f'{name}.log'}
        manifest['commands'].append(record);save()
        try:
            with (args.out/f'{name}.log').open('w',encoding='utf-8',errors='replace') as stream:
                proc=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
            record['exit_code']=proc.returncode
            save()
            if proc.returncode != 0:
                print(f'[FAIL] {name} exit code {proc.returncode}. Log: {args.out/(name+".log")}')
                return proc.returncode or 1
        except Exception as e:
            record['error']=str(e);save();raise
        print('[OK]',name,flush=True)
    try:
        summary=build_evidence(args.out,args.samples,set(groups_list),steps=args.steps,
                               seed=args.seed,bootstrap=args.bootstrap)
    except Exception as e:
        manifest['analysis_error']=str(e);save();raise
    manifest['complete']=True
    manifest['pairwise_rows']=len(summary['paired'])
    save()
    print('[DONE] Verified output: ',args.out/'evidence_report.md')
    print('[DONE] Machine-readable:',args.out/'paired_comparisons.csv')
    print('[CAUTION] Six-pose exploratory bootstrap, NOT independent statistical significance.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
