"""Reproducible *synthetic-signal* contact detector ablation.

This measures robustness to deliberate contact dropout / phantom geometry; it
is NOT a MuJoCo grasp benchmark or evidence from independent human subjects.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.contact_detection import ContactDetectionConfig, ForceContactDebouncer

METHODS = ('geometry_only', 'force_only', 'temporal_only', 'force_and_temporal')


def synth_episode(rng: np.random.Generator, frames: int = 240) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Known truth with noisy solver-like force plus false geometric contacts."""
    start = int(rng.integers(35, 60))
    stop = int(rng.integers(165, 195))
    truth = np.zeros(frames, dtype=bool)
    truth[start:stop] = True
    force = np.zeros(frames, dtype=float)
    # True contact: a different force amplitude per episode, random dropouts.
    typical_force = float(rng.uniform(0.025, 0.07))
    force[truth] = np.clip(rng.normal(typical_force, typical_force*.17, size=truth.sum()), 0, None)
    missing = (rng.random(frames) < 0.13) & truth
    force[missing] = 0.0
    # Inactive/phantom contacts reported by a geometric detector: 0-3 mN.
    ghost = (~truth) & (rng.random(frames) < 0.19)
    force[ghost] = rng.uniform(0.0001, 0.0025, size=ghost.sum())
    # Rare spikes with a transient force, not sustained gripping.
    glitch = (~truth) & (rng.random(frames) < 0.035)
    force[glitch] = rng.uniform(0.012, 0.026, size=glitch.sum())
    # The geometry-only baseline also accepts solver-inactive candidates.
    candidates = (force > 0) | ((~truth) & (rng.random(frames) < 0.11))
    return truth, force, candidates


def predict(force: np.ndarray, candidates: np.ndarray, cfg: ContactDetectionConfig) -> dict[str,np.ndarray]:
    baseline_temporal = ForceContactDebouncer(ContactDetectionConfig(
        enter_force_n=0.00001, exit_force_n=0.000001,
        enter_frames=cfg.enter_frames, exit_frames=cfg.exit_frames))
    proposed = ForceContactDebouncer(cfg)
    n = len(force)
    results = {name: np.zeros(n, dtype=bool) for name in METHODS}
    results['geometry_only'][:] = candidates
    results['force_only'][:] = force >= cfg.enter_force_n
    for i, value in enumerate(force):
        results['temporal_only'][i] = baseline_temporal.update({'finger': float(value)})['finger']
        results['force_and_temporal'][i] = proposed.update({'finger': float(value)})['finger']
    return results


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str,float]:
    tp = int(np.sum(y & pred))
    fp = int(np.sum(~y & pred))
    fn = int(np.sum(y & ~pred))
    tn = int(np.sum(~y & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    onset = int(np.flatnonzero(y)[0])
    candidates = np.flatnonzero(pred[onset:])
    delay = int(candidates[0]) if len(candidates) else len(y) - onset
    return {'precision': precision, 'recall': recall, 'f1':f1,
            'fpr':fp/(fp+tn) if fp+tn else 0.0,
            'fnr':fn/(fn+tp) if fn+tp else 0.0,
            'state_switches': int(np.count_nonzero(np.diff(pred.astype(int)))),
            'onset_delay_frames':delay}


def run(outdir: Path, *, episodes:int=40, seed:int=20261108, bootstraps:int=2000,
        config: ContactDetectionConfig|None=None) -> dict:
    if episodes < 2 or bootstraps < 100: raise ValueError('episodes>=2 and bootstraps>=100 required')
    cfg = config or ContactDetectionConfig()
    rng = np.random.default_rng(seed)
    rows=[]
    for episode in range(episodes):
        truth, force, candidates = synth_episode(rng)
        preds = predict(force, candidates, cfg)
        for name, pred in preds.items():
            rows.append({'episode':episode,'method':name,**metrics(truth,pred)})
    outdir.mkdir(parents=True,exist_ok=True)
    with (outdir/'contact_synthetic_episodes.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    brng=np.random.default_rng(seed+99)
    summary={}
    for method in METHODS:
        subset=[r for r in rows if r['method']==method]
        summary[method]={metric:float(np.mean([r[metric] for r in subset]))
                         for metric in ('precision','recall','f1','fpr','fnr','state_switches','onset_delay_frames')}
    # paired episode bootstrap, not independent-frame bootstrap
    differences={}
    for competitor in METHODS[:-1]:
        delta=np.array([next(r['f1'] for r in rows if r['episode']==ep and r['method']=='force_and_temporal')
                        - next(r['f1'] for r in rows if r['episode']==ep and r['method']==competitor)
                        for ep in range(episodes)],dtype=float)
        draw=brng.integers(0,episodes,size=(bootstraps,episodes))
        means=np.mean(delta[draw],axis=1)
        differences[competitor]={'mean_paired_f1_delta':float(np.mean(delta)),
                                  'bootstrap_95ci':list(map(float,np.quantile(means,[.025,.975])))}
    result={'schema':'synthetic_contact_detector_ablation_v1',
            'warning':'Engineered synthetic force/noise; no MuJoCo physics or tactile ground truth; repeated episodes share simulation model.',
            'seed':seed,'episodes':episodes,'frames_per_episode':240,
            'config':vars(cfg),'summary':summary,
            'paired_episode_bootstrap_f1_delta_vs_proposed':differences}
    (outdir/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--outdir',type=Path,default=Path('experiments/实验09_StageC五方法MuJoCo对比/接触判定消融/results/generated'))
    p.add_argument('--episodes',type=int,default=40)
    p.add_argument('--seed',type=int,default=20261108)
    args=p.parse_args(argv)
    result=run(args.outdir,episodes=args.episodes,seed=args.seed)
    for method, row in result['summary'].items():
        print(f"{method:20} F1={row['f1']:.3f} FPR={row['fpr']:.3f} switches={row['state_switches']:.1f}")
    print('[CAUTION] Synthetic-signal diagnostics; not measured grasp improvement.')
    print('[DONE]',args.outdir)


if __name__=='__main__':
    main()
