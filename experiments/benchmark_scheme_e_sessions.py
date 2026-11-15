"""Cross-recording Scheme-E replay on captured CH-M6 sessions.

This is a held-out recording replay with frozen mapper/configs. Target labels are
preset pose-table targets; no independent subject IDs are present. It is not a
subject-independent or hardware validation study.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re, statistics, sys, time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.实验08_任务效应谱方案对比.benchmark_scheme_e_research import run as run_scheme_e
from src.glove.captured_samples import load_captured_samples


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]):
    cols = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def parse_session(path: Path):
    name = path.parent.name
    m = re.match(r'(?P<date>\d{8})_(?P<time>\d{6})_(?P<serial>.+)_(?P<side>left|right)$', name)
    if not m:
        return {'session_id': name, 'glove_serial': 'unknown', 'hand_side': 'unknown'}
    return {'session_id': name, 'glove_serial': m.group('serial'), 'hand_side': m.group('side')}


def read_csv_rows(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def parse_scalar(value):
    if value in ('', None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sessions-root', type=Path, default=ROOT / '数据手套数据/records/chm6_calibrations')
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/scheme_e_cross_recording')
    ap.add_argument('--modes', default='E_dynamic,E_fixed6', help='controlled comparison; default dynamic vs fixed k=6')
    ap.add_argument('--deadline-ms', type=float, default=1000 / 60)
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261109)
    ap.add_argument('--process-timeout', type=float, default=90.0)
    ap.add_argument('--max-sessions', type=int, default=0, help='0 means all selected valid recordings')
    ap.add_argument('--start-index', type=int, default=1, help='1-based start position in sorted valid recording list')
    ap.add_argument('--resume', action='store_true', help='reuse complete sessions already present under output/sessions')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)

    modes = args.modes.split(',')
    allowed = {'E_dynamic', 'E_fixed4', 'E_fixed5', 'E_fixed6'}
    if modes != list(dict.fromkeys(modes)) or any(m not in allowed for m in modes):
        ap.error('modes must be a unique subset of E_dynamic,E_fixed4,E_fixed5,E_fixed6')
    if args.start_index < 1:
        ap.error('--start-index must be >= 1')
    if args.process_timeout <= 0:
        ap.error('--process-timeout must be positive')

    files = sorted(args.sessions_root.glob('*/captured_samples.csv'))
    all_valid = []
    skipped = []
    for p in files:
        try:
            samples = load_captured_samples(p)
            if any(x.targets_deg is None for x in samples) or len({x.pose for x in samples}) < 2:
                raise ValueError('missing target labels or fewer than two pose clusters')
            meta = parse_session(p)
            meta.update({'path': str(p), 'sha256': sha256(p), 'frames': len(samples),
                         'pose_clusters': len({x.pose for x in samples}),
                         'pose_sequence_count': len({x.pose_sequence_index for x in samples})})
            all_valid.append((p, samples, meta))
        except Exception as exc:
            skipped.append({'path': str(p), 'reason': str(exc)})
    selected = all_valid[args.start_index - 1:]
    if args.max_sessions:
        selected = selected[:args.max_sessions]
    if not all_valid or not selected:
        ap.error(f'no valid sessions found under {args.sessions_root} for requested selection')

    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema': 'scheme_e_cross_recording_v2',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'modes': modes,
        'deadline_ms': args.deadline_ms,
        'recordings': [m for _, _, m in all_valid],
        'skipped': skipped,
        'selection': {'start_index_1based': args.start_index, 'max_sessions': args.max_sessions,
                      'selected_session_ids': [m['session_id'] for _, _, m in selected]},
        'scope': 'held-out recording replay with fixed released mapper; labels are preset pose-table targets; not independent subjects or physical grasp proof',
        'complete': False,
    }
    if args.dry_run:
        print('[DRY RUN] valid recordings:', len(all_valid), 'selected:', len(selected), 'modes:', ','.join(modes))
        for _, _, x in selected:
            print('[SESSION]', x['session_id'], x['glove_serial'], x['hand_side'], x['frames'], x['pose_clusters'], x['sha256'])
        print('[SKIPPED]', len(skipped))
        return 0

    (args.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    execution_failures = []
    reused = []
    for idx, (p, samples, meta) in enumerate(selected, args.start_index):
        session_id = meta['session_id']
        out = args.out / 'sessions' / session_id
        expected = set(modes)
        complete_existing = False
        if args.resume and (out / 'run_summary.csv').is_file() and (out / 'per_frame_metrics.csv').is_file():
            try:
                existing = read_csv_rows(out / 'run_summary.csv')
                existing_modes = {r.get('policy') for r in existing}
                frame_rows = read_csv_rows(out / 'per_frame_metrics.csv')
                complete_existing = expected.issubset(existing_modes) and all(
                    sum(1 for r in frame_rows if r.get('policy') == mode) == len(samples) for mode in modes)
            except Exception:
                complete_existing = False
        if complete_existing:
            print(f'[REUSE {idx}/{len(all_valid)}] {session_id}', flush=True)
            reused.append(session_id)
            continue

        ns = SimpleNamespace(samples=p, steps=len(samples), modes=','.join(modes), repeats=1,
                             bootstrap=300, seed=args.seed + idx, source='sensor', out=out,
                             process_timeout=args.process_timeout, deadline_ms=args.deadline_ms, overwrite=True)
        print(f'[SESSION {idx}/{len(all_valid)}] {session_id} n={len(samples)} poses={meta["pose_clusters"]}', flush=True)
        try:
            run_scheme_e(ns)
        except Exception as exc:
            execution_failures.append({'session_id': session_id, 'path': str(p), 'error': repr(exc)})
            print(f'[FAILED {idx}/{len(all_valid)}] {session_id}: {exc}', flush=True)

    # Aggregate *all completed session folders*, not only the subset processed
    # in this invocation. This makes --start-index + --resume safe for batching.
    session_rows = []
    pose_rows = []
    final_failures = list(execution_failures)
    fail_by_id = {x['session_id']: x for x in final_failures}
    for p, samples, meta in all_valid:
        out = args.out / 'sessions' / meta['session_id']
        summary_path = out / 'run_summary.csv'
        frame_path = out / 'per_frame_metrics.csv'
        if not summary_path.is_file() or not frame_path.is_file():
            if meta['session_id'] not in fail_by_id:
                final_failures.append({'session_id': meta['session_id'], 'path': str(p), 'error': 'missing complete run_summary.csv/per_frame_metrics.csv'})
            continue
        try:
            rr = read_csv_rows(summary_path)
            fm = read_csv_rows(frame_path)
            if not set(modes).issubset({r.get('policy') for r in rr}):
                raise ValueError('one or more requested methods missing from run_summary.csv')
            for mode in modes:
                n = sum(1 for r in fm if r.get('policy') == mode)
                if n != len(samples):
                    raise ValueError(f'{mode}: expected {len(samples)} frame rows, got {n}')
            for row in rr:
                if row.get('policy') not in modes:
                    continue
                session_rows.append({**{k: meta[k] for k in ('session_id', 'glove_serial', 'hand_side')},
                    'frames': len(samples), 'pose_clusters': meta['pose_clusters'], 'data_sha256': meta['sha256'],
                    **{k: parse_scalar(v) for k, v in row.items() if k not in {'policy', 'csv_path', 'log_path'}},
                    'policy': row['policy']})
            for mode in modes:
                for pose in sorted({r['pose'] for r in fm if r.get('policy') == mode}):
                    vals = [float(r['teacher_mae_deg']) for r in fm if r.get('policy') == mode and r.get('pose') == pose]
                    pose_rows.append({**{k: meta[k] for k in ('session_id', 'glove_serial', 'hand_side')},
                                      'policy': mode, 'pose': pose, 'frame_count': len(vals),
                                      'mae_deg': statistics.mean(vals)})
        except Exception as exc:
            if meta['session_id'] not in fail_by_id:
                final_failures.append({'session_id': meta['session_id'], 'path': str(p), 'error': f'invalid output: {exc}'})

    write_csv(args.out / 'session_summary.csv', session_rows)
    write_csv(args.out / 'pose_summary.csv', pose_rows)
    by_method = defaultdict(list)
    for r in session_rows:
        by_method[r['policy']].append(r)
    method_rows = []
    fields = ['teacher_label_mae_deg', 'teacher_label_rmse_deg', 'optimizer_median_ms', 'optimizer_p95_ms',
              'pipeline_p95_ms', 'pipeline_p99_ms', 'deadline_miss_rate', 'within_sequence_command_step_rms_deg',
              'within_sequence_command_velocity_rms_deg_s', 'requested_k_mean', 'actual_output_low_rank_fraction',
              'accepted_fallback_rate']
    for mode, rr in by_method.items():
        outrow = {'policy': mode, 'recording_count': len(rr),
                  'unique_glove_serial_hand_pairs': len({(r['glove_serial'], r['hand_side']) for r in rr})}
        for field in fields:
            vals = [float(r[field]) for r in rr if r.get(field) not in ('', None) and r.get(field) is not None]
            outrow[field + '_mean_across_recordings'] = statistics.mean(vals) if vals else None
            outrow[field + '_sd_across_recordings'] = statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else None)
        method_rows.append(outrow)
    write_csv(args.out / 'method_summary.csv', method_rows)

    lookup = {(r['session_id'], r['policy']): r for r in session_rows}
    pairs = []
    if 'E_dynamic' in modes and 'E_fixed6' in modes:
        for r in session_rows:
            if r['policy'] != 'E_dynamic':
                continue
            b = lookup.get((r['session_id'], 'E_fixed6'))
            if b:
                pairs.append({'session_id': r['session_id'], 'glove_serial': r['glove_serial'], 'hand_side': r['hand_side'],
                    'dynamic_minus_fixed6_mae_deg': float(r['teacher_label_mae_deg']) - float(b['teacher_label_mae_deg']),
                    'dynamic_minus_fixed6_pipeline_p95_ms': float(r['pipeline_p95_ms']) - float(b['pipeline_p95_ms']),
                    'dynamic_minus_fixed6_deadline_miss_rate': float(r['deadline_miss_rate']) - float(b['deadline_miss_rate'])})
    write_csv(args.out / 'paired_recording_deltas.csv', pairs)

    # Aggregate actual requested/accepted/output dimensions and fallback causes.
    dimension_counts = defaultdict(int)
    fallback_counts = defaultdict(int)
    rejection_totals = defaultdict(int)
    for _, _, meta in all_valid:
        mode_file = args.out / 'sessions' / meta['session_id'] / 'mode_selection.csv'
        if not mode_file.is_file():
            continue
        for row in read_csv_rows(mode_file):
            policy = row.get('policy', '')
            if policy not in modes:
                continue
            for key in ('requested_k', 'accepted_k', 'output_rank'):
                try:
                    v = int(round(float(row.get(key, 'nan'))))
                except (TypeError, ValueError):
                    continue
                dimension_counts[(policy, key, v)] += 1
            try:
                if float(row.get('fallback', 0) or 0) > 0:
                    fallback_counts[(policy, row.get('fallback_reason') or 'unspecified')] += 1
            except (TypeError, ValueError):
                pass
            for key in ('reject_spectral_count', 'reject_fidelity_count', 'reject_nonlinear_count'):
                try:
                    rejection_totals[(policy, key)] += int(float(row.get(key, 0) or 0))
                except (TypeError, ValueError):
                    pass
    dimension_rows = [{'policy': k[0], 'dimension_field': k[1], 'dimension': k[2], 'frame_count': v}
                      for k, v in sorted(dimension_counts.items())]
    write_csv(args.out / 'dimension_summary.csv', dimension_rows)
    fallback_rows = [{'policy': k[0], 'fallback_reason': k[1], 'frame_count': v}
                     for k, v in sorted(fallback_counts.items())]
    write_csv(args.out / 'fallback_reason_summary.csv', fallback_rows)
    rejection_rows = [{'policy': k[0], 'rejection_type': k[1], 'total_candidate_rejections': v}
                      for k, v in sorted(rejection_totals.items())]
    write_csv(args.out / 'candidate_rejection_summary.csv', rejection_rows)

    # Pair records within the same glove serial + hand side to avoid treating
    # repeated recordings from one device-hand combination as independent units.
    pair_groups = defaultdict(list)
    for row in pairs:
        pair_groups[(row['glove_serial'], row['hand_side'])].append(row)
    device_pair_rows = []
    for (serial, side), group in sorted(pair_groups.items()):
        device_pair_rows.append({
            'glove_serial': serial, 'hand_side': side, 'recording_count': len(group),
            'dynamic_minus_fixed6_mae_deg_mean': statistics.mean(float(x['dynamic_minus_fixed6_mae_deg']) for x in group),
            'dynamic_minus_fixed6_pipeline_p95_ms_mean': statistics.mean(float(x['dynamic_minus_fixed6_pipeline_p95_ms']) for x in group),
            'dynamic_minus_fixed6_deadline_miss_rate_mean': statistics.mean(float(x['dynamic_minus_fixed6_deadline_miss_rate']) for x in group),
        })
    write_csv(args.out / 'paired_device_hand_deltas.csv', device_pair_rows)
    pair_mae = [float(r['dynamic_minus_fixed6_mae_deg_mean']) for r in device_pair_rows]
    pair_p95 = [float(r['dynamic_minus_fixed6_pipeline_p95_ms_mean']) for r in device_pair_rows]
    pair_stats = {
        'device_hand_group_count': len(device_pair_rows),
        'dynamic_minus_fixed6_mae_deg_mean_across_device_hand_groups': statistics.mean(pair_mae) if pair_mae else None,
        'dynamic_minus_fixed6_mae_deg_sd_across_device_hand_groups': statistics.stdev(pair_mae) if len(pair_mae) > 1 else None,
        'dynamic_better_group_count': sum(x < 0 for x in pair_mae),
        'fixed6_better_group_count': sum(x > 0 for x in pair_mae),
        'tie_group_count': sum(x == 0 for x in pair_mae),
        'dynamic_minus_fixed6_pipeline_p95_ms_mean_across_device_hand_groups': statistics.mean(pair_p95) if pair_p95 else None,
        'interpretation': 'device-hand grouping only; not independent subject-level inference',
    }
    (args.out / 'paired_device_hand_summary.json').write_text(json.dumps(pair_stats, ensure_ascii=False, indent=2), encoding='utf-8')

    processed = sorted({r['session_id'] for r in session_rows})
    all_pair_keys = {(x['glove_serial'], x['hand_side']) for x in [m for _, _, m in all_valid]}
    notes = [
        '# 方案 E 跨采集记录评估', '',
        f'- 扫描有效记录数：{len(all_valid)}；完整成功记录数：{len(processed)}；失败/缺失记录数：{len(final_failures)}。',
        f'- 不同手套序列号/左右手组合：{len(all_pair_keys)}。',
        '- 每个 captured_samples.csv 是完整留出评估记录；运行期间不使用该记录拟合映射参数。',
        '- 记录来自预设姿态采集；target_1..6 为预设姿态表目标，不是独立测量的真实机械手关节角。',
        '- 项目没有独立受试者 ID，因此不能声称独立受试者泛化；重复手套序列号在多个日期可能重复。',
        '- within-sequence step/velocity 仅统计同一 pose_sequence_index 内的稳定保持序列，不代表自由手势连续动作。',
        f'- 实时预算：{args.deadline_ms:.3f} ms；deadline miss 是记录到的 pipeline_latency_ms 超预算，不是硬实时保证。',
        '', '## 各方法跨记录汇总', '',
        '| 方法 | 记录数 | MAE 均值° | MAE 跨记录 SD° | Pipeline P95 均值 ms | 60Hz 超时比例 | 序列内命令步长 RMS° | 实际低秩输出率 | fallback 率 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in method_rows:
        def fmt(k, d=3):
            v = r.get(k + '_mean_across_recordings')
            return '—' if v is None else f'{float(v):.{d}f}'
        sd = r.get('teacher_label_mae_deg_sd_across_recordings')
        notes.append(f"| {r['policy']} | {r['recording_count']} | {fmt('teacher_label_mae_deg')} | {'—' if sd is None else f'{float(sd):.3f}'} | {fmt('pipeline_p95_ms',2)} | {fmt('deadline_miss_rate',3)} | {fmt('within_sequence_command_step_rms_deg')} | {fmt('actual_output_low_rank_fraction',3)} | {fmt('accepted_fallback_rate',3)} |")
    notes += ['', '## 配对记录结果', '', '| 记录 | Dynamic − Fixed6 MAE° | Pipeline P95 差 ms | Deadline miss 差 |', '|---|---:|---:|---:|']
    for r in pairs:
        notes.append(f"| {r['session_id']} | {r['dynamic_minus_fixed6_mae_deg']:+.3f} | {r['dynamic_minus_fixed6_pipeline_p95_ms']:+.2f} | {r['dynamic_minus_fixed6_deadline_miss_rate']:+.3f} |")
    notes += ['', '## 设备-手别分组配对摘要', '',
              f'- 配对 MAE 差均值（Dynamic − Fixed6）：{pair_stats["dynamic_minus_fixed6_mae_deg_mean_across_device_hand_groups"]:.4f}°（{pair_stats["device_hand_group_count"]} 个设备-手别组合；不是受试者数）。',
              f'- Dynamic 较低的组合：{pair_stats["dynamic_better_group_count"]}；Fixed6 较低的组合：{pair_stats["fixed6_better_group_count"]}。',
              '- 分组结果用于检查重复记录是否由少数手套主导；不能用于独立受试者显著性推断。',
              '', '## 动态维度与回退统计', '',
              f'- 维度分布表：`dimension_summary.csv`；fallback 原因：`fallback_reason_summary.csv`；候选拒绝统计：`candidate_rejection_summary.csv`。',
              f'- 同一手套序列号/左右手内配对汇总：`paired_device_hand_deltas.csv`（{len(device_pair_rows)} 个设备-手别组合）。',
              '', '## 失败记录', '']
    notes += [f"- {f['session_id']}: {f['error']}" for f in final_failures] if final_failures else ['- 无。']
    notes += ['', '## 解释边界', '',
        '- 这是跨采集记录回放，不是独立受试者研究；相同序列号的多次记录不能作为独立个体数。',
        '- 每个姿态保持序列存在静态目标，序列内平滑性指标不等价于自由手势连续动作。',
        '- 动态 E 与固定 k=6 使用同一个 E 求解链路，仅改变维数规则，是主要受控比较。',
        '- 本结果不证明真实机械手跟踪准确度、抓握成功或部署安全。', '']
    (args.out / 'report.md').write_text('\n'.join(notes), encoding='utf-8')

    manifest['processed_recordings'] = processed
    manifest['reused_recordings'] = reused
    manifest['failures'] = final_failures
    manifest['complete'] = len(final_failures) == 0 and len(processed) == len(all_valid)
    manifest['status'] = 'complete' if manifest['complete'] else 'partial_with_failures'
    manifest['output_files'] = ['session_summary.csv', 'pose_summary.csv', 'method_summary.csv', 'paired_recording_deltas.csv', 'paired_device_hand_deltas.csv', 'paired_device_hand_summary.json', 'dimension_summary.csv', 'fallback_reason_summary.csv', 'candidate_rejection_summary.csv', 'report.md']
    manifest['aggregation_scope'] = 'aggregates every valid recording with complete per-session outputs under output/sessions, including sessions completed in earlier batches'
    (args.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[DONE]', args.out / 'report.md', 'processed=', len(processed), 'failures=', len(final_failures))
    return 0 if not final_failures else 2


if __name__ == '__main__':
    raise SystemExit(main())
