# 实采手套映射的 MuJoCo 配对实验

运行：

```powershell
.venv\Scripts\python.exe experiments\compare_glove_tasks.py
```

可用 `--samples`、`--base-mapping`、`--manifest` 指定交付文件；`--steps`（默认 240）、`--warmup-frames`（默认 30）和 `--output-dir` 控制回放及输出。需要安装 MuJoCo。脚本独立运行，不改变 `src.main --input sim` 或 `--input glove` 的接口。

协议：PINCH 使用实采“张开”和“对食指”，WRAP 使用“张开”和“握拳”。各姿态只使用第 4 帧留出样本；前 30 帧保持张开，后 210 帧保持任务姿态。两种映射使用相同的传感器输入、任务场景、控制频率和下游控制流程，每次试验重新建立 MuJoCo 场景。base 输出由交付的 `base_mapping.json` 计算；personalized 输出由 `manifest.json` 对应留出帧的 `final_prediction_deg` 提供。交付包没有 `.pt` 权重，故这里是留出姿态的静态重复回放，不是连续动作的个性化模型在线推理。重复帧不能视为独立样本。

默认写入 `outputs/glove_tasks/`：

- `summary.csv`：每个任务 × 映射一行，含活动帧物体目标点到指尖的平均距离误差 `fingertip_distance_error_mean_m`、人手目标指尖跟踪误差、接触帧比例 `contact_success_rate`、稳定接触帧比例 `stable_contact_ratio`、连续接触成功布尔值、物体最大位移、关节跟踪 RMSE、留出样本映射 RMSE 和处理延迟。
- `frames.csv`：逐帧输入样本、阶段和上述可逐帧统计的指标，供绘图与核查。
- `summary.json`：相同的四组汇总、输入路径、帧数和实验限制。

接触与稳定指标来自现有仿真代理判据。PINCH 的接触成功可能包含接近阶段的固定物体帧；稳定接触只统计释放后的自由物体帧。此实验不代表真实物体抓取或不同用户的统计成功率。
