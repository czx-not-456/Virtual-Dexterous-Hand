#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage C pipeline audit: paired methods within the SAME sensor condition.

Usage from CH-M6 Stage C project root:
  python CHM6_StageC_Control_Path_Audit.py --outputs outputs

Read only. Requires original Stage-C run CSV files with --servo-trace.
Reports per-stage max and mean absolute differences in *degrees*.
"""
import argparse
import csv
import math
from pathlib import Path
from statistics import fmean

FINGERS = {
    "Thumb": ("th_cmc_joint", "th_mcp_joint", "th_ip_joint"),
    "Index": ("ff_mcp_joint", "ff_dip_joint"),
    "Middle": ("mf_mcp_joint", "mf_dip_joint"),
    "Ring": ("rf_mcp_joint", "rf_dip_joint"),
    "Little": ("lf_mcp_joint", "lf_dip_joint"),
}
TASK_JOINTS = FINGERS["Thumb"] + FINGERS["Index"]
JOINTS = tuple(n for names in FINGERS.values() for n in names)
PAIRINGS = (
    ("robust6", "stagec-alpha", "alpha 相对 robust6"),
    ("stagec-beta", "stagec-dual", "dual 相对 beta"),
    ("robust6", "stagec-beta", "beta 相对 robust6"),
    ("robust6", "robust4", "robust4 相对 robust6"),
)
STAGES = ("opt", "servo", "filtered", "command", "actual")
SCALE = 180.0 / math.pi


def file_for(root, condition, method):
    name = f"stageC_pinch_{condition}_{method}.csv"
    found = list(root.rglob(name))
    if len(found) != 1:
        raise ValueError(f"应找到唯一文件 {name}，实际 {len(found)}：{found}")
    return found[0]


def load(path):
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        data = {}
        for row in reader:
            frame = int(row["frame"])
            if frame in data:
                raise ValueError(f"重复帧 {frame}：{path}")
            data[frame] = row
    if not data:
        raise ValueError(f"文件为空：{path}")
    required = [f"trace_{s}_{n}" for s in STAGES[:-1] for n in TASK_JOINTS]
    required += [f"q_{n}" for n in JOINTS]
    missing = [col for col in required if col not in next(iter(data.values()))]
    if missing:
        raise ValueError(f"{path.name} 缺失字段 {missing[:6]}；请确认运行时有 --servo-trace")
    return data


def changes(a, b, frame_list, stage, joints):
    diffs = []
    for frame in frame_list:
        ra, rb = a[frame], b[frame]
        for joint in joints:
            col = f"q_{joint}" if stage == "actual" else f"trace_{stage}_{joint}"
            x, y = ra[col], rb[col]
            if x is None or y is None or not x.strip() or not y.strip():
                raise ValueError(f"空值：frame={frame}, col={col}")
            xv, yv = float(x), float(y)
            if not (math.isfinite(xv) and math.isfinite(yv)):
                raise ValueError(f"非有限数据：frame={frame}, col={col}")
            diffs.append(abs(xv-yv)*SCALE)
    return (max(diffs), fmean(diffs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--start", type=int, default=120)
    parser.add_argument("--stop", type=int, default=240)
    args = parser.parse_args()
    if args.start >= args.stop:
        parser.error("--start 必须小于 --stop")
    print("STAGE C 控制链路对照：同一输入条件下比较两个方法（单位：度）")
    print("opt=映射后 | servo=Hold后 | filtered=滤波后 | command=下发指令 | actual=MuJoCo实际位置")
    print("仅检测同帧差异；未校正控制方法之间可能存在的历史状态差异。\n")
    cache = {}
    for condition in ("clean", "mask_three"):
        print(f"{'='*75}\n条件: {condition} | 统计窗口 [{args.start},{args.stop})")
        for a_method, b_method, title in PAIRINGS:
            for method in (a_method, b_method):
                key = condition, method
                if key not in cache:
                    path = file_for(args.outputs, condition, method)
                    cache[key] = load(path)
            a = cache[condition, a_method]
            b = cache[condition, b_method]
            frames = list(range(args.start, args.stop))
            for frame in frames:
                if frame not in a or frame not in b:
                    raise ValueError(f"{condition} {title} 缺少帧 {frame}")
            print(f"\n{title}")
            for stage in STAGES:
                selected = JOINTS if stage == "actual" else TASK_JOINTS
                mx, avg = changes(a, b, frames, stage, selected)
                print(f"  {stage:8s}  max={mx:.9f}° mean={avg:.9f}° ({len(selected)} joints)")
            amx, _ = changes(a, b, frames, "actual", FINGERS["Thumb"] + FINGERS["Index"])
            mmx, _ = changes(a, b, frames, "actual", FINGERS["Middle"])
            print(f"  actual-by-digit: thumb+index max={amx:.9f}°, middle max={mmx:.9f}°")
    print("\n重要：如果 opt 非零但 servo/command/actual 为零，说明修正在下游未形成关节差异。")
    print("若 alpha 与 robust6 在 actual 一致，仅能证明该实验下的最终关节响应一致；")
    print("不能据此证明 alpha 的映射层计算没有运行，或推广到其他抓握任务。")


if __name__ == '__main__':
    main()
