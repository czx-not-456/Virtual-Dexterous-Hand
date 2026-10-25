# Virtual-Dexterous-Hand 测试报告

日期：2026-10-08；代码基线：`91229f22038217202dd801f00c3fc32f23992196`；当前配置：`ch-m6-adapter-v1`。目录名 v100 按用户要求归档。

## 1. 测试环境与方法

Windows；Python 3.11.9；NumPy 2.4.6；SciPy 1.17.1；PyYAML 6.0.3；MuJoCo 3.13.0；pytest 9.1.1。

使用项目 `.venv/Scripts/python.exe`，无界面、非实时节流运行。各实验顺序执行，避免本次测试彼此争用 CPU；延迟仍受本机负载影响。未安装或升级依赖。所有运行命令保存于 [execution_summary.json](execution_summary.json)，代码、模型、数据哈希和环境见 [environment.json](environment.json)。

## 2. 命令执行与自动化测试

| 检查 | 退出码 | 总耗时 s | 原始日志 |
| --- | --- | --- | --- |
| 01_pip_check | 0 | 4.444 | [01_pip_check.log](01_pip_check.log) |
| 02_pytest | 0 | 183.458 | [02_pytest.log](02_pytest.log) |
| 03_mapping | 0 | 14.157 | [03_mapping.log](03_mapping.log) |
| 04_glove_calibration | 0 | 0.55 | [04_glove_calibration.log](04_glove_calibration.log) |
| 05_dataset_pipeline | 0 | 50.514 | [05_dataset_pipeline.log](05_dataset_pipeline.log) |
| 06_mock_pipeline | 0 | 38.476 | [06_mock_pipeline.log](06_mock_pipeline.log) |
| 07_glove_pipeline | 0 | 1.805 | [07_glove_pipeline.log](07_glove_pipeline.log) |
| 08_synthetic_benchmark | 0 | 429.284 | [08_synthetic_benchmark.log](08_synthetic_benchmark.log) |
| 09_glove_tasks | 0 | 28.322 | [09_glove_tasks.log](09_glove_tasks.log) |

自动化测试：**52 passed / 0 failed或error / 0 skipped**。覆盖标定、数据加载、特征与意图、映射限位、模型构建、物理时钟、任务伺服、评估语义、benchmark 错误处理及三类任务的 800 帧 MuJoCo 集成流程。JUnit 见 [pytest_results.xml](pytest_results.xml)。

## 3. 三任务合成数据 MuJoCo benchmark

协议：trial 0、1、2；neutral、pinch、wrap 各完整 800 帧，共 **9 次／7200 帧**。未用截断数据。程序执行失败 **0** 次；任务规定指标未通过 **0** 次。

| 任务 | trial 数 | 规定指标通过率 | 接触成功率 | 稳定成功率 | 活动帧接触比例均值 | 活动帧稳定比例均值 | P95 延迟均值 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 3 | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 123.06 |
| pinch | 3 | 100.00% | 100.00% | 0.00% | 99.15% | 3.53% | 116.21 |
| wrap | 3 | 100.00% | 100.00% | 100.00% | 93.03% | 9.76% | 119.36 |

neutral 只要求流程完成，其接触／稳定率为零是预期结果。PINCH 必需指标为 contact，WRAP 为 stable。成功要求连续 12 个活动帧（60 Hz 配置下约 0.2 秒）；stable 同时约束物体姿态漂移 ≤5°，只统计释放后的自由物体。接触帧比例以活动帧为分母；表中的成功率以所选 trial 数为分母，不能作为真实用户总体成功概率。

本次各运行 P95 管线延迟范围为 **109.91–132.80 ms**，60 Hz 单帧预算为 16.67 ms。因此本次结果不支持严格 60 Hz 实时执行；物理时钟同步和实际计算吞吐应分别评价。

详细结果： [benchmark_runs.csv](benchmark/benchmark_runs.csv)、[benchmark_summary.csv](benchmark/benchmark_summary.csv)、[BENCHMARK_REPORT.md](benchmark/BENCHMARK_REPORT.md)、[HTML 看板](benchmark/benchmark_dashboard.html)。每次运行的 CSV 和 summary JSON 在同目录。

## 4. 静态／动态动作映射

Mock 对照脚本每种姿态计算 80 次。PINCH 位置 MSE：0.02339360 → 0.01612208，降低 31.08%；PINCH 向量 MSE：0.00319963 → 0.00352732，增加 10.24%；WRAP 向量 MSE：0.00172031 → 0.00171467，降低 0.33%。当前 `synergies: {}`，耦合惩罚均为零，不能据此声称耦合改善。各目标变化不一致，不能概括为所有误差均下降。原始输出见 [03_mapping.log](03_mapping.log)。

## 5. 实采标定误差对照

交付数据为 24 帧、6 种姿态，每姿态第 4 帧留出；本次复算 6 个样本 × 6 个输出。base RMSE **16.2204°**；personalized manifest 预测 RMSE **0.1606°**；降低 **99.01%**。

本次读取交付预测，未重新训练或加载个性化模型。样本少且来自固定标定姿态，不能推断跨用户或连续动作泛化。结果见 [JSON](glove_calibration/mapping_comparison.json) 和 [逐姿态 CSV](glove_calibration/mapping_comparison_by_pose.csv)。

## 6. 实采姿态 MuJoCo 配对实验

协议：PINCH／WRAP × base／personalized 共 4 组，每组 240 帧，前 30 帧张开、后 210 帧固定任务姿态，总计 960 帧。两种映射使用相同留出帧及下游控制。

| 任务 | 映射 | 目标点距离误差 mm | 接触帧比例 | 稳定帧比例 | 连续接触成功 | 连续稳定成功 | P95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pinch | base | 5.124 | 50.48% | 0.00% | True | False | 17.84 |
| pinch | personalized | 8.030 | 56.19% | 0.00% | True | False | 12.97 |
| wrap | base | 6.382 | 89.52% | 89.52% | True | True | 49.22 |
| wrap | personalized | 7.423 | 77.62% | 77.62% | True | True | 56.98 |

误差列是物体目标点到机器人指尖的平均距离，和标定角度 RMSE 是不同指标。个性化预测的角度误差改善不保证任务接触／稳定指标同步改善。本次 WRAP 的个性化映射稳定帧比例由基础映射的 89.52% 降至 77.62%；两个任务的物体目标点距离误差也均增大。重复帧不构成独立受试样本；PINCH 接触可包含接近夹具帧；这些结果不代表真实抓起、搬运或放置。

结果见 [summary.csv](glove_tasks/summary.csv)、[summary.json](glove_tasks/summary.json)、[frames.csv](glove_tasks/frames.csv)。

## 7. 三种输入链路与结果完整性

dataset 800 帧、Mock 800 帧、实采 glove 24 帧的主程序无物理后端运行均完成，逐帧 CSV 和 summary JSON 位于 [input_checks](input_checks/)。此处验证输入／算法／日志链路，不以接触任务指标为验收条件。无物理后端时 contact/stable 为零，glove 的 24 帧顺序也不保证激活 PINCH。

另行核对了 9 组 benchmark 的逐帧数量与 JSON、一致的 11 DoF／11 actuator、非零任务活动帧、960 行配对回放结果、无物理后端的零接触以及汇总数值的有限性。已审查的源码、配置、原始模型及输入数据哈希均保持一致。完整性检查：**通过**，见 [artifact_verification.json](artifact_verification.json)。

## 8. 未覆盖范围与复现

未操作真实手套硬件，未执行个性化模型在线推理，未人工检查 MuJoCo Viewer 交互，未验证 Unity 接收端；无抬升、搬运、放置或跨用户实验。当前架构和历史文档差异见 [REVIEW_NOTES.md](REVIEW_NOTES.md)。本次只更新进度简报并生成测试归档。

在项目根目录复现本次测试：

```powershell
.\.venv\Scripts\python.exe test_results_v100_20261008\run_validation.py
.\.venv\Scripts\python.exe test_results_v100_20261008\finalize_report.py
```

重新运行会更新本目录测试文件及 pytest 临时目录；请先复制本次归档以保留历史证据。
