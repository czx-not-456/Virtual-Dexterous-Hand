# -*- coding: utf-8 -*-
"""CH-M6 Stage C fingertip target-error audit: Hold vs original, same 120..239 window.

Usage from project root:
 python CHM6_StageC_Precision_Compare.py --outputs ./outputs

Pure Python standard library; never modifies experiments or trained models.
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
from statistics import mean

METHODS = ('robust6', 'robust4', 'stagec-alpha', 'stagec-beta', 'stagec-dual')
COLS = ('servo_thumb_error_m', 'servo_index_error_m')


def path_single(root: Path, filename: str) -> Path | None:
    matches = sorted(root.rglob(filename))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f'Found multiple files named {filename}: {matches}')
    return matches[0]


def safe_num(text: str | None, path: Path, frame: int, col: str) -> float:
    try:
        v = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{path}, frame={frame}, invalid {col}={text!r}') from exc
    if not math.isfinite(v) or v < 0:
        raise ValueError(f'{path}, frame={frame}, expected nonnegative finite {col}; got {v}')
    return v * 1000.0


def percentile(values: list[float], prob: float) -> float:
    s = sorted(values)
    idx = (len(s) - 1) * prob
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    return s[lo] * (hi - idx) + s[hi] * (idx - lo) if hi > lo else s[lo]


def summarize(path: Path, start: int, stop: int) -> dict:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f'CSV no header: {path}')
        missing = [x for x in ('frame', *COLS) if x not in reader.fieldnames]
        if missing:
            raise ValueError(f'CSV missing fields {missing}: {path}')
        by_frame = {}
        for row in reader:
            frame = int(row['frame'])
            if frame in by_frame:
                raise ValueError(f'Duplicate frame={frame}: {path}')
            if start <= frame < stop:
                by_frame[frame] = row
    required = set(range(start, stop))
    if set(by_frame) != required:
        missing = sorted(required - set(by_frame))
        raise ValueError(f'{path}: expected {stop-start} frames [{start},{stop}), missing={missing[:10]}')
    thumb, index = [], []
    for frame in range(start, stop):
        thumb.append(safe_num(by_frame[frame]['servo_thumb_error_m'], path, frame, 'servo_thumb_error_m'))
        index.append(safe_num(by_frame[frame]['servo_index_error_m'], path, frame, 'servo_index_error_m'))
    both = thumb + index
    return {
        'n_frames': stop - start,
        'thumb_mean_mm': mean(thumb),
        'index_mean_mm': mean(index),
        'both_mean_mm': mean(both),
        'both_rmse_mm': math.sqrt(mean([v*v for v in both])),
        'thumb_p95_mm': percentile(thumb, .95),
        'index_p95_mm': percentile(index, .95),
        'thumb_max_mm': max(thumb),
        'index_max_mm': max(index),
        'path': str(path.resolve()),
    }


def fmt(x: float) -> str:
    return f'{x:.4f}'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outputs', type=Path, default=Path('./outputs'))
    ap.add_argument('--start', type=int, default=120)
    ap.add_argument('--stop', type=int, default=240)
    args = ap.parse_args()
    root = args.outputs.resolve()
    if not root.is_dir() or args.stop <= args.start:
        ap.error('outputs must be a directory and start < stop')
    runs = {}
    for method in METHODS:
        for condition in ('clean', 'mask_three'):
            filename = f'stageC_pinch_{condition}_{method}.csv'
            fp = path_single(root, filename)
            if fp is not None:
                runs[('hold', condition, method)] = summarize(fp, args.start, args.stop)
    for method, stem in [('robust6', 'original_mask_three_robust6'),
                         ('stagec-alpha', 'original_mask_three_alpha')]:
        fp = path_single(root, stem + '.csv')
        if fp is not None:
            runs[('original', 'mask_three', method)] = summarize(fp, args.start, args.stop)
    if not runs:
        raise SystemExit(f'No Stage C result CSV found in {root}. No accuracy calculation performed.')

    print(f'实验精度：物体任务目标到指尖的伺服空间距离；单位mm（越小越好）')
    print(f'同一统计窗口：[ {args.start}, {args.stop} )，每次 {args.stop-args.start} 帧')
    print('指标均值由CSV原始每帧servo_thumb/index_error_m计算；不是软件教师标签MAE。')
    print('控制器   传感器条件  映射方法        拇指均值   食指均值   双指均值    双指RMSE   拇指P95    食指P95')
    for (servo, cond, method), d in sorted(runs.items(), key=lambda x:(x[0][0]!='hold',x[0][1]!='clean',METHODS.index(x[0][2]))):
        print(f'{servo:10} {cond:11} {method:15} {fmt(d["thumb_mean_mm"]):>9} {fmt(d["index_mean_mm"]):>10} {fmt(d["both_mean_mm"]):>10} {fmt(d["both_rmse_mm"]):>11} {fmt(d["thumb_p95_mm"]):>10} {fmt(d["index_p95_mm"]):>10}')

    print('\n同控制器、同故障条件下，相对于robust6的双指均值改善：')
    for servo in ('hold','original'):
        base = runs.get((servo,'mask_three','robust6'))
        if base is None:
            continue
        for method in METHODS:
            r = runs.get((servo,'mask_three',method))
            if r is None: continue
            diff = base['both_mean_mm'] - r['both_mean_mm']
            pct = (diff / base['both_mean_mm'] * 100) if base['both_mean_mm'] else float('nan')
            print(f'  {servo:8} {method:15} 减少 {diff:+.4f} mm，改善 {pct:+.3f}%')

    print('\n同映射方法，关闭新增Hold(original)相对于Hold的误差变化：')
    for method in ('robust6','stagec-alpha'):
        hold = runs.get(('hold','mask_three',method))
        original = runs.get(('original','mask_three',method))
        if not (hold and original):
            print(f'  {method:15}: 无法比较（缺少对应original或hold CSV）')
            continue
        v = original['both_mean_mm'] - hold['both_mean_mm']
        print(f'  {method:15}: original - hold = {v:+.4f} mm（正值表示关闭Hold后误差更大）')

    a = runs.get(('original','mask_three','stagec-alpha'))
    b = runs.get(('original','mask_three','robust6'))
    if a and b:
        print('\n关闭新增Hold后 α-only 相比 robust6：')
        for key, label in [('thumb_mean_mm','拇指'),('index_mean_mm','食指'),('both_mean_mm','双指')]:
            v = b[key] - a[key]
            print(f'  {label:5} α-only减少 {v:+.4f} mm（正值是α更优）')
    else:
        print('\n[注意] 关闭Hold的两组CSV尚未齐备，无法计算original条件下α的精度收益。')
    out = root / 'stageC_precision_comparison.csv'
    with out.open('w', encoding='utf-8-sig', newline='') as f:
        keys = ('servo','condition','method','n_frames','thumb_mean_mm','index_mean_mm','both_mean_mm','both_rmse_mm','thumb_p95_mm','index_p95_mm','thumb_max_mm','index_max_mm','path')
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for (servo,condition,method), value in sorted(runs.items()):
            writer.writerow({'servo': servo,'condition':condition,'method':method,**value})
    print(f'\n[DONE] 汇总: {out}')
    print('[限制] 同一手套片段的确定性回放；未形成独立试验置信区间或物理抓握精度证明。')


if __name__ == '__main__':
    main()
