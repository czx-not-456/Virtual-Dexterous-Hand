#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-C joint audit, stdlib only. Read existing MuJoCo run CSVs; never rerun or modify simulations.

Run in the project root:
    python CHM6_StageC_Fault_Joint_Audit.py --outputs outputs
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import math
from pathlib import Path
from statistics import fmean

METHODS = ('robust6', 'robust4', 'stagec-alpha', 'stagec-beta', 'stagec-dual')
FINGERS = {
    'Thumb': ('q_th_cmc_joint', 'q_th_mcp_joint', 'q_th_ip_joint'),
    'Index': ('q_ff_mcp_joint', 'q_ff_dip_joint'),
    'Middle': ('q_mf_mcp_joint', 'q_mf_dip_joint'),
    'Ring': ('q_rf_mcp_joint', 'q_rf_dip_joint'),
    'Little': ('q_lf_mcp_joint', 'q_lf_dip_joint'),
}
JOINTS = tuple(joint for cols in FINGERS.values() for joint in cols)
DEG_PER_RAD = 180 / math.pi


def locate(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f'应当找到唯一CSV: {filename}; 实际找到 {len(matches)} 个: {matches}')
    return matches[0]


def read_csv(path: Path) -> dict[int, dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        required = ('frame',) + JOINTS
        missing = [col for col in required if col not in columns]
        if missing:
            raise RuntimeError(f'{path} 缺失列: {missing}; q_*示例: {[c for c in columns if c.startswith("q_")]}')
        by_frame = {}
        for row in reader:
            try:
                f = int(row['frame'])
            except (ValueError, TypeError, KeyError) as err:
                raise RuntimeError(f'{path}: 无效frame: {row.get("frame")!r}') from err
            if f in by_frame:
                raise RuntimeError(f'{path}: 重复frame={f}')
            for joint in JOINTS:
                try:
                    value = float(row[joint])
                except (ValueError, TypeError, KeyError) as err:
                    raise RuntimeError(f'{path}: frame={f}, {joint} 无效: {row.get(joint)!r}') from err
                if not math.isfinite(value):
                    raise RuntimeError(f'{path}: frame={f}, {joint} 非有限数: {value!r}')
            by_frame[f] = row
        if not by_frame:
            raise RuntimeError(f'{path}: CSV无有效数据')
        return by_frame


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def fault_window_assertions(fault: dict[int, dict[str, str]], start: int, stop: int) -> None:
    required = [100, 119, start, min(start + 30, stop - 1), stop - 1, stop, stop + 10]
    # print status; these checks diagnose, do not require every pre/post checkpoint for arbitrary windows.
    print('  故障标记抽查:')
    for frame in dict.fromkeys(required):
        if frame not in fault:
            print(f'    帧{frame}: CSV中缺失')
            continue
        row = fault[frame]
        print(f'    帧{frame}: active={row.get("stagec_fault_active", "<无列>")}, '
              f'count={row.get("stagec_fault_count", "<无列>")}, '
              f'invalid={row.get("mapping_invalid_channels", "<无列>")}')


def main() -> None:
    parser = argparse.ArgumentParser(description='CH-M6 Stage C faulty-vs-clean real joint-position audit')
    parser.add_argument('--outputs', type=Path, default=Path('outputs'))
    parser.add_argument('--start', type=int, default=120)
    parser.add_argument('--stop', type=int, default=240, help='不包含此帧')
    parser.add_argument('--save', type=Path, default=Path('outputs/stageC_joint_fault_audit.csv'))
    args = parser.parse_args()
    if args.stop <= args.start:
        parser.error('--stop必须大于--start')
    root = args.outputs.resolve()
    if not root.is_dir():
        raise RuntimeError(f'输出目录不存在: {root}')
    rows_out = []
    overall_out = []
    print('Stage C: 正常与三路故障运行的实际11关节位置对照（单位：度）')
    print(f'故障窗口: [{args.start}, {args.stop})，期望 {args.stop - args.start} 帧')
    print('注：q_* 由工程的 q_actual（MuJoCo 实际关节位置）写入；不是控制目标。')

    for method in METHODS:
        clean_path = locate(root, f'stageC_pinch_clean_{method}.csv')
        fault_path = locate(root, f'stageC_pinch_mask_three_{method}.csv')
        clean = read_csv(clean_path)
        fault = read_csv(fault_path)
        valid_frames = list(range(args.start, args.stop))
        missing_frames = [f for f in valid_frames if f not in clean or f not in fault]
        if missing_frames:
            raise RuntimeError(f'{method}: 缺少{len(missing_frames)}帧，例如{missing_frames[:8]}')

        print(f'\n[{method}]')
        print(f'  正常: {clean_path}  SHA256前12位={sha(clean_path)}')
        print(f'  故障: {fault_path}  SHA256前12位={sha(fault_path)}')
        if method == 'stagec-dual':
            fault_window_assertions(fault, args.start, args.stop)

        joint_stats = []
        max_detail = (0.0, None, None, None, None)
        for finger, joints in FINGERS.items():
            diffs = []
            for frame in valid_frames:
                for joint in joints:
                    a, b = float(clean[frame][joint]), float(fault[frame][joint])
                    diff = abs(b - a) * DEG_PER_RAD
                    diffs.append(diff)
                    if diff > max_detail[0]:
                        max_detail = (diff, frame, joint, a, b)
            entry = dict(method=method, finger=finger,
                         joint_count=len(joints), frame_count=len(valid_frames),
                         max_difference_deg=max(diffs),
                         mean_difference_deg=fmean(diffs))
            joint_stats.append(entry)
            rows_out.append(entry)

        for item in joint_stats:
            print(f'  {item["finger"]:<7} 最大差={item["max_difference_deg"]:.9f}°, '
                  f'平均绝对差={item["mean_difference_deg"]:.9f}°')
        maximum, frame, joint, a, b = max_detail
        overall_out.append((method, maximum))
        print(f'  所有11关节最大差={maximum:.9f}°')
        if joint is not None:
            print(f'  最大差所在行: frame={frame}, joint={joint}, '
                  f'正常q={a:.12g} rad, 故障q={b:.12g} rad')
        else:
            # The previous PowerShell script found nonzero values; explicitly flag discrepancy.
            print('  [警告] 读取的q_*在120~239帧完全相同。检查上述CSV路径和SHA；可能是实验文件被覆盖或之前统计脚本有误。')
            joint = JOINTS[0]
            for f in (args.start, min(args.start + 30, args.stop - 1), args.stop - 1):
                print(f'  验证原始值 frame={f}, {joint}: '
                      f'clean={clean[f][joint]!r}, fault={fault[f][joint]!r}')

    save = args.save.resolve()
    save.parent.mkdir(parents=True, exist_ok=True)
    with save.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    print('\n' + '=' * 70)
    print('最终交叉核对（应与先前MaxJointDiffDeg接近，除非CSV已变化）：')
    for method, maximum in overall_out:
        print(f'  {method:<13}: {maximum:.9f}°')
    print(f'[DONE] 已输出: {save}')

if __name__ == '__main__':
    main()
