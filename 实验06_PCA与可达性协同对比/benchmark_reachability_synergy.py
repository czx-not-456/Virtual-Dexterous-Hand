"""Reproducible non-contact diagnostic: fixed PCA vs scheme B vs reachability-gated scheme C.

Runs identical glove replay through the full command pipeline in --sim none,
then computes six-command errors relative to *preset teacher pose labels*.
Do NOT interpret this as grasp success, measured robot error, or real-time speed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.glove.captured_samples import load_captured_samples
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.latent_synergy import LatentSynergyMapper


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def run_one(mode, samples, args, out):
    name = f'diagnostic_{mode}'
    command = [sys.executable, '-m', 'src.main', '--sim', 'none', '--input', 'glove',
               '--glove-samples', str(samples), '--steps', str(args.steps),
               '--latent-model', 'configs/latent_chm6_adaptive.json',
               '--latent-source', args.source, '--output-dir', str(out), '--run-name', name]
    if mode == 'adaptive_B':
        command += ['--adaptive-synergy']
    if mode in ('reachability_C', 'reachability_no_residual', 'reachability_fixed6'):
        command += ['--reachability-synergy']
        configs = {
            'reachability_C': 'configs/reachability_synergy.yaml',
            'reachability_no_residual': 'configs/reachability_no_residual.yaml',
            'reachability_fixed6': 'configs/reachability_fixed6.yaml',
        }
        command += ['--reachability-config', configs[mode]]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return read_csv(out / (name + '.csv'))


def measures(rows, captures, mapping):
    q = np.array([[float(row['q_' + joint]) for joint in mapping.robot.joint_order] for row in rows])
    refs = np.array([r.targets_deg for r in captures[:len(rows)]])
    if not all(r.targets_deg is not None for r in captures[:len(rows)]):
        raise ValueError('Diagnostic requires teacher target labels')
    active = [mapping.teacher.index[name] for name in mapping.teacher.active_order]
    controls_deg = np.rad2deg(q[:, active])
    coupling_err = max(float(np.max(np.abs(
        q[:, mapping.teacher.index[child]] - gain * q[:, mapping.teacher.index[driver]])))
        for driver, child, gain in mapping.teacher.couplings)
    out = {
        'frames': len(rows),
        'teacher_label_mae_deg': float(np.abs(controls_deg - refs).mean()),
        'teacher_label_mae_per_control_deg': np.abs(controls_deg - refs).mean(axis=0).round(5).tolist(),
        'max_software_coupling_error_rad': coupling_err,
        'joint_range_violations': int(np.sum((q < mapping.robot.ranges[:, 0]-1e-8) |
                                             (q > mapping.robot.ranges[:, 1]+1e-8))),
        'optimizer_median_ms': float(np.median([float(r['optimizer_ms']) for r in rows])),
        'pipeline_p95_ms': float(np.percentile([float(r['pipeline_latency_ms']) for r in rows],95)),
    }
    if np.isfinite(float(rows[0].get('adaptive_k','nan'))):
        out['dynamic_k_histogram'] = {str(i): sum(int(float(r['adaptive_k'])) == i for r in rows)
                                       for i in range(1, 7) if sum(int(float(r['adaptive_k'])) == i for r in rows)}
        out['mean_residual_coefficient_norm'] = float(np.mean([
            float(r['adaptive_residual_norm']) for r in rows]))
        out['optimizer_feasible_fraction'] = float(np.mean([
            float(r['adaptive_feasible']) for r in rows]))
    if 'reach_variables' in rows[0] and np.isfinite(float(rows[0]['reach_variables'])):
        variables = [int(float(r['reach_variables'])) for r in rows]
        out['true_optimization_variables_mean'] = float(np.mean(variables))
        out['true_optimization_variables_histogram'] = {
            str(k): variables.count(k) for k in sorted(set(variables))}
        out['fallback_used_fraction'] = float(np.mean([float(r.get('reach_fallback_used', 0)) for r in rows]))
        out['fraction_residual_activated'] = float(np.mean([
            float(r['reach_residual_axes']) > 0 for r in rows]))
        out['task_linear_error_mean_m_equivalent'] = float(np.mean([
            float(r['reach_task_linear_error_m']) for r in rows]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, default=ROOT / '数据手套数据/captured_samples(1).xlsx')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'results/generated/reachability_diagnostic')
    parser.add_argument('--source', choices=['sensor','control','blend'], default='sensor')
    parser.add_argument('--steps', type=int, default=24)
    args = parser.parse_args()
    args.samples = args.samples.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    captures = load_captured_samples(args.samples)
    if args.steps > len(captures) or args.steps <= 0:
        raise ValueError('steps must be positive and <= number of input samples (avoid looped data)')
    mapping = LatentSynergyMapper(RobotHandModel(), ROOT / 'configs/latent_chm6_adaptive.json',
                                  ROOT / '数据手套数据/mod/joint_mapping.json')
    result = {'data': str(args.samples), 'reference_type': 'six predefined pose labels, not joint ground truth',
              'contact_or_grasp_metric': 'not measured', 'input_source':args.source,
              'runs': {}}
    for mode in ('fixed_PCA_k4', 'adaptive_B', 'reachability_fixed6',
                 'reachability_no_residual', 'reachability_C'):
        rows = run_one(mode, args.samples, args, args.out)
        result['runs'][mode] = measures(rows, captures, mapping)
    path = args.out / 'comparison_summary.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('Caution: this diagnostic contains NO physical contact or hardware measurements.')


if __name__ == '__main__':
    main()
