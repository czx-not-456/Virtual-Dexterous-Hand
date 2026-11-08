"""Summarize completed checks and verify saved experiment artifacts."""
from pathlib import Path
import os
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))
def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))
def truth(value):
    return str(value).lower() in ("true", "1")
def pct(value):
    return f"{100 * float(value):.2f}%"
def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |", *["| " + " | ".join(map(str, row)) + " |" for row in rows]])
execution = read_json(OUT / "execution_summary.json")
if len(execution) != 9:
    raise RuntimeError("Validation runner has not finished all nine checks")
environment = read_json(OUT / "environment.json")
junit = ET.parse(OUT / "pytest_results.xml").getroot()
suites = list(junit.iter("testsuite"))
tests = sum(int(s.get("tests", 0)) for s in suites)
failed = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
skipped = sum(int(s.get("skipped", 0)) for s in suites)
benchmark = read_csv(OUT / "benchmark" / "benchmark_runs.csv")
aggregates = read_csv(OUT / "benchmark" / "benchmark_summary.csv")
GLOVE_TASKS = OUT / "glove_tasks"
if not GLOVE_TASKS.exists():
    GLOVE_TASKS = ROOT / "experiments" / "实验04_手套抓握任务对比" / "results"
GLOVE_CALIBRATION = OUT / "glove_calibration"
if not GLOVE_CALIBRATION.exists():
    GLOVE_CALIBRATION = ROOT / "experiments" / "实验03_手套标定映射对比" / "results"
glove = read_json(GLOVE_TASKS / "summary.json")["trials"]
calibration = read_json(GLOVE_CALIBRATION / "mapping_comparison.json")
issues = []
def check(condition, message):
    if not condition:
        issues.append(message)
check(all(r["returncode"] == 0 for r in execution), "At least one executed command did not succeed")
check(len(benchmark) == 9, "Expected nine synthetic benchmark runs")
check(len(glove) == 4, "Expected four captured-glove paired trials")
summaries = sorted((OUT / "benchmark").glob("*_summary.json"))
check(len(summaries) == 9, "Expected nine synthetic summary JSON files")
def finite(value, path):
    if isinstance(value, float):
        check(math.isfinite(value), f"Nonfinite number: {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            finite(child, path + "." + key)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            finite(child, path + f"[{i}]")
for path in summaries:
    summary = read_json(path)
    finite(summary, path.name)
    rows = read_csv(path.with_name(path.name.replace("_summary.json", ".csv")))
    check(len(rows) == summary["total_frames"] == 800, "Frame count mismatch: " + path.name)
    check(summary["robot_dof"] == summary["robot_actuators"] == 11, "Architecture mismatch: " + path.name)
    check(summary["active_frames"] > 0, "No active task frames: " + path.name)
for name, expected in [("dataset", 800), ("mock", 800), ("glove", 24)]:
    summary = read_json(OUT / "input_checks" / (name + "_summary.json"))
    finite(summary, name)
    rows = read_csv(OUT / "input_checks" / (name + ".csv"))
    check(len(rows) == summary["total_frames"] == expected, "Input frame count mismatch: " + name)
    check(summary["contact_frames"] == summary["stable_frames"] == 0, "Nonphysics run reported contact: " + name)
frames = read_csv(GLOVE_TASKS / "frames.csv")
check(len(frames) == 960, "Expected 960 paired-glove frame rows")
for trial in glove:
    finite(trial, trial["task"] + "/" + trial["mapping"])
    check(trial["frames"] == 240 and trial["active_frames"] == 210, "Unexpected paired-glove protocol frame counts")
changed = []
for path, digest in environment["sha256"].items():
    if not (ROOT / path).is_file() or hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
        changed.append(path)
check(not changed, "Reviewed source/config/model/data changed during tests: " + ", ".join(changed))
command_failures = sum(not truth(r.get("run_ok")) for r in benchmark)
metric_failures = sum(truth(r.get("run_ok")) and not truth(r.get("task_metric_pass")) for r in benchmark)
bench_table = table(["任务", "trial 数", "规定指标通过率", "接触成功率", "稳定成功率", "活动帧接触比例均值", "活动帧稳定比例均值", "P95 延迟均值 ms"],
    [[r["task"], r["trials"], pct(r["task_pass_rate"]), pct(r["contact_success_rate"]), pct(r["stable_success_rate"]), pct(r["mean_contact_ratio"]), pct(r["mean_stable_ratio"]), f'{float(r["mean_p95_latency_ms"]):.2f}'] for r in aggregates])
glove_table = table(["任务", "映射", "目标点距离误差 mm", "接触帧比例", "稳定帧比例", "连续接触成功", "连续稳定成功", "P95 ms"],
    [[r["task"], r["mapping"], f'{r["fingertip_distance_error_mean_m"] * 1000:.3f}', pct(r["contact_success_rate"]), pct(r["stable_contact_ratio"]), r["contact_success_proxy"], r["stable_success_proxy"], f'{r["p95_pipeline_latency_ms"]:.2f}'] for r in glove])
latencies = [read_json(p)["p95_pipeline_latency_ms"] for p in summaries]
verified = {"checked_at": datetime.now().astimezone().isoformat(), "artifact_checks_pass": not issues,
            "issues": issues, "reviewed_files_unchanged": not changed, "pytest_tests": tests, "pytest_failures": failed,
            "pytest_skipped": skipped, "benchmark_runs": len(benchmark), "benchmark_command_failures": command_failures,
            "benchmark_task_metric_failures": metric_failures, "paired_glove_trials": len(glove),
            "benchmark_p95_latency_ms_range": [min(latencies), max(latencies)]}
(OUT / "artifact_verification.json").write_text(json.dumps(verified, ensure_ascii=False, indent=2), encoding="utf-8")
glove_tasks_link = Path(os.path.relpath(GLOVE_TASKS, OUT)).as_posix()
glove_calibration_link = Path(os.path.relpath(GLOVE_CALIBRATION, OUT)).as_posix()
command_table = table(["检查", "退出码", "总耗时 s", "原始日志"], [[r["name"], r["returncode"], r["elapsed_seconds"], f'[{r["name"]}.log]({r["name"]}.log)'] for r in execution])
report = f'''# Virtual-Dexterous-Hand 测试报告

日期：2026-10-08；代码基线：`{environment["commit"]}`；当前配置：`ch-m6-adapter-v1`。目录名 v100 按用户要求归档。

## 1. 测试环境与方法

Windows；Python {environment["python"].split()[0]}；NumPy {environment["packages"]["numpy"]}；SciPy {environment["packages"]["scipy"]}；PyYAML {environment["packages"]["PyYAML"]}；MuJoCo {environment["packages"]["mujoco"]}；pytest {environment["packages"]["pytest"]}。

使用项目 `.venv/Scripts/python.exe`，无界面、非实时节流运行。各实验顺序执行，避免本次测试彼此争用 CPU；延迟仍受本机负载影响。未安装或升级依赖。所有运行命令保存于 [execution_summary.json](execution_summary.json)，代码、模型、数据哈希和环境见 [environment.json](environment.json)。

## 2. 命令执行与自动化测试

{command_table}

自动化测试：**{tests - failed - skipped} passed / {failed} failed或error / {skipped} skipped**。覆盖标定、数据加载、特征与意图、映射限位、模型构建、物理时钟、任务伺服、评估语义、benchmark 错误处理及三类任务的 800 帧 MuJoCo 集成流程。JUnit 见 [pytest_results.xml](pytest_results.xml)。

## 3. 三任务合成数据 MuJoCo benchmark

协议：trial 0、1、2；neutral、pinch、wrap 各完整 800 帧，共 **9 次／7200 帧**。未用截断数据。程序执行失败 **{command_failures}** 次；任务规定指标未通过 **{metric_failures}** 次。

{bench_table}

neutral 只要求流程完成，其接触／稳定率为零是预期结果。PINCH 必需指标为 contact，WRAP 为 stable。成功要求连续 12 个活动帧（60 Hz 配置下约 0.2 秒）；stable 同时约束物体姿态漂移 ≤5°，只统计释放后的自由物体。接触帧比例以活动帧为分母；表中的成功率以所选 trial 数为分母，不能作为真实用户总体成功概率。

本次各运行 P95 管线延迟范围为 **{min(latencies):.2f}–{max(latencies):.2f} ms**，60 Hz 单帧预算为 16.67 ms。因此本次结果不支持严格 60 Hz 实时执行；物理时钟同步和实际计算吞吐应分别评价。

详细结果： [benchmark_runs.csv](benchmark/benchmark_runs.csv)、[benchmark_summary.csv](benchmark/benchmark_summary.csv)、[BENCHMARK_REPORT.md](benchmark/BENCHMARK_REPORT.md)、[HTML 看板](benchmark/benchmark_dashboard.html)。每次运行的 CSV 和 summary JSON 在同目录。

## 4. 静态／动态动作映射

Mock 对照脚本每种姿态计算 80 次。PINCH 位置 MSE：0.02339360 → 0.01612208，降低 31.08%；PINCH 向量 MSE：0.00319963 → 0.00352732，增加 10.24%；WRAP 向量 MSE：0.00172031 → 0.00171467，降低 0.33%。当前 `synergies: {{}}`，耦合惩罚均为零，不能据此声称耦合改善。各目标变化不一致，不能概括为所有误差均下降。原始输出见 [03_mapping.log](03_mapping.log)。

## 5. 实采标定误差对照

交付数据为 24 帧、6 种姿态，每姿态第 4 帧留出；本次复算 6 个样本 × 6 个输出。base RMSE **{calibration["base_mapping"]["rmse_deg"]:.4f}°**；personalized manifest 预测 RMSE **{calibration["personalized_mapping"]["rmse_deg"]:.4f}°**；降低 **{calibration["rmse_improvement_percent"]:.2f}%**。

本次读取交付预测，未重新训练或加载个性化模型。样本少且来自固定标定姿态，不能推断跨用户或连续动作泛化。结果见 [JSON]({glove_calibration_link}/mapping_comparison.json) 和 [逐姿态 CSV]({glove_calibration_link}/mapping_comparison_by_pose.csv)。

## 6. 实采姿态 MuJoCo 配对实验

协议：PINCH／WRAP × base／personalized 共 4 组，每组 240 帧，前 30 帧张开、后 210 帧固定任务姿态，总计 960 帧。两种映射使用相同留出帧及下游控制。

{glove_table}

误差列是物体目标点到机器人指尖的平均距离，和标定角度 RMSE 是不同指标。个性化预测的角度误差改善不保证任务接触／稳定指标同步改善。本次 WRAP 的个性化映射稳定帧比例由基础映射的 89.52% 降至 77.62%；两个任务的物体目标点距离误差也均增大。重复帧不构成独立受试样本；PINCH 接触可包含接近夹具帧；这些结果不代表真实抓起、搬运或放置。

结果见 [summary.csv]({glove_tasks_link}/summary.csv)、[summary.json]({glove_tasks_link}/summary.json)、[frames.csv]({glove_tasks_link}/frames.csv)。

## 7. 三种输入链路与结果完整性

dataset 800 帧、Mock 800 帧、实采 glove 24 帧的主程序无物理后端运行均完成，逐帧 CSV 和 summary JSON 位于 [input_checks](input_checks/)。此处验证输入／算法／日志链路，不以接触任务指标为验收条件。无物理后端时 contact/stable 为零，glove 的 24 帧顺序也不保证激活 PINCH。

另行核对了 9 组 benchmark 的逐帧数量与 JSON、一致的 11 DoF／11 actuator、非零任务活动帧、960 行配对回放结果、无物理后端的零接触以及汇总数值的有限性。已审查的源码、配置、原始模型及输入数据哈希均保持一致。完整性检查：**{"通过" if not issues else "未通过"}**，见 [artifact_verification.json](artifact_verification.json)。

## 8. 未覆盖范围与复现

未操作真实手套硬件，未执行个性化模型在线推理，未人工检查 MuJoCo Viewer 交互，未验证 Unity 接收端；无抬升、搬运、放置或跨用户实验。当前架构和历史文档差异见 [REVIEW_NOTES.md](REVIEW_NOTES.md)。本次只更新进度简报并生成测试归档。

在项目根目录复现本次测试：

```powershell
.\\.venv\\Scripts\\python.exe test\\test_results_v100_20261008\\run_validation.py
.\\.venv\\Scripts\\python.exe test\\test_results_v100_20261008\\finalize_report.py
```

重新运行会更新本目录测试文件及 pytest 临时目录；请先复制本次归档以保留历史证据。
'''
(OUT / "TEST_REPORT.md").write_text(report, encoding="utf-8")
progress = ROOT / "项目进度报告相关文件" / "项目进度简报_20261008.md"
text = progress.read_text(encoding="utf-8").split("\n## 七、本次批量与配对测试结果")[0]
text += f'''\n\n## 七、本次批量与配对测试结果

合成数据选取 trial 0–2，每个任务完整 800 帧，共 9 次／7200 帧；命令失败 {command_failures} 次，规定任务指标失败 {metric_failures} 次。

{bench_table}

各运行 P95 管线延迟为 {min(latencies):.2f}–{max(latencies):.2f} ms，超过 60 Hz 的 16.67 ms 单帧预算，当前仍需性能优化。

实采固定姿态回放的四组配对结果如下；每组 240 帧，其中 210 帧为活动姿态。

{glove_table}

接触成功和稳定成功必须分开解读。本次个性化映射虽降低标定角度误差，但 WRAP 稳定帧比例从 89.52% 降至 77.62%，两个任务的物体目标点距离误差均增大，后续需要联合评估标定精度和任务表现。完整日志、逐帧数据、判据与限制见 [测试报告](../test/test_results_v100_20261008/TEST_REPORT.md)。
'''
progress.write_text(text, encoding="utf-8")
print(json.dumps(verified, ensure_ascii=False, indent=2))
if issues:
    raise SystemExit(1)
