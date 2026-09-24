# VALIDATION

本文档汇总项目的历史版本验证记录。测试日期、通过数量和实验数值仅代表对应版本及当时环境，不代表当前代码状态。
## CH-M6 当前版本（2026-09-23）

统一模型为 `CH-M6/CH-M6_L.xml`，项目侧通过运行时适配器注入 STL assets、指尖 site、接触 geom、桌面和任务物体，第三方目录保持只读。

```powershell
pytest -p no:cacheprovider -rA
```

结果：`48 passed in 171.35s`。全量测试包含 800 帧 neutral、PINCH、WRAP 真实 MuJoCo 试验：neutral 流程通过；PINCH 连续双指腹接触通过；WRAP 连续接触与稳定接触通过。

当前架构：`CH-M6_L, nominal DoF=11, effective actuators=11`。模拟手套和人体特征仍使用 15 个标准语义通道，随后由配置映射为 CH-M6 11 个物理关节。

待指导老师确认的机械参数见 `configs/robot_hand.yaml` 中 `TODO(指导老师)`。

## v0.3

生成环境验证日期：2026-09-11

### 1. 自动化测试

执行：

```bash
pytest -rA
```

结果：

```text
15 passed, 1 skipped
```

跳过项为 MuJoCo runtime schema 测试，因为生成环境未安装 `mujoco`。用户本机已经能够运行 MuJoCo，因此该项应在本机真实执行。

### 2. 纯算法闭环

执行：

```bash
python -m src.main --sim none --steps 900
```

确认能够依次识别：

```text
NEUTRAL -> PINCH -> NEUTRAL -> WRAP -> NEUTRAL
```

并显示：

```text
Robot architecture: nominal DoF=15, effective actuators=7
```

### 3. v0.3 Mock 对照实验

执行：

```bash
python experiments/compare_mapping.py
```

本次工程参数下得到：

```text
PINCH position MSE improvement: 9.58%
PINCH vector MSE improvement:   40.03%
WRAP coupling penalty improvement: 83.64%
WRAP vector MSE improvement: -41.77%
```

最后一项为负并非自动判定失败：WRAP 模式提高了欠驱协同约束权重，会牺牲一部分“与人手相对向量完全一致”的目标。这恰好体现多目标映射的任务权衡。

以上均为 Mock 工程验证数据，不能直接作为结题论文的正式实验结论。

### 4. 本机需继续验证

```powershell
pytest -rA
python -m src.main --sim mujoco --render --steps 0 --realtime --task wrap
python -m src.main --sim mujoco --render --steps 0 --realtime --task pinch --camera closeup
python -m src.main --sim mujoco --render --steps 0 --realtime --task sphere
```

重点检查：

- 7 个 actuator 的 MuJoCo schema 是否正常加载；
- 非拇指 tendon + spring 是否产生合理屈曲；
- PINCH 时拇指与食指是否可对薄块形成接触；
- WRAP 时 thumb/fingers/palm 接触区是否逐渐增加；
- `outputs/run_*.csv` 中接触与姿态指标是否正常变化。

## v0.4.0

在项目根目录执行：

```powershell
.\.venv\Scripts\Activate.ps1
pytest -rA
```

然后依次执行：

```powershell
python -m src.main --sim mujoco --input dataset --dataset-trial 0 --task pinch --steps 600
python -m src.main --sim mujoco --input dataset --dataset-trial 0 --task wrap --steps 600
python -m src.main --sim mujoco --input dataset --dataset-trial 0 --task sphere --steps 600
```

优先检查 `*_summary.json`：

- `success_proxy`
- `contact_frame_ratio`
- `stable_frame_ratio`
- `max_success_streak_frames`
- `first_contact_latency_ms`
- `p95_pipeline_latency_ms`
- `max_orientation_drift_deg`
- `mean_servo_error_m`

PINCH 目标：相较 v0.3.4 的 `pinch_contact_proxy=0`，至少应首先出现 thumb/index 同时指腹接触。若仍为 0，保留 CSV，不要继续凭观察调参，优先根据 `servo_thumb_error_m` / `servo_index_error_m` 与接触位置定位问题。

批量实验：

```powershell
python experiments\benchmark_synthetic.py --trials 5 --steps 600
```

最终用于大创阶段汇报的表格建议直接取 `outputs/benchmark/benchmark_summary.csv`，可视化演示使用 `benchmark_dashboard.html`。
