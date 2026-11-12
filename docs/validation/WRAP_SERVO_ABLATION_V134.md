# WRAP 多指协调接触控制：四方法消融（候选 v1.3.4）

> **状态：已实现，等待独立 MuJoCo 实验验证。** 这是基于 v1.3.3 项目新增的控制模块和批量实验入口，**不是**已经证明有效的新方法。所有稳定性指标仍为仿真代理指标；不代表真实机械手完成了实物抓握。

## 1. 为什么改

用户在 v1.3.3 的同一合成手套回放（trial 0）中观察到圆柱被手指推动而移动，且有自动复位现象。已完成的原先实验显示（历史数据，不是新方案结果）：

| WRAP 原策略 | 接触率 | 最大相对平移跨度 | 最大相对旋转跨度 | 结束时严格稳定 |
| --- | ---: | ---: | ---: | --- |
| 几何接触 + 原伺服增益 0.80 | 92.9% | 6.63 mm | 1.17° | False |
| 几何接触 + 低伺服增益 0.40 | 94.1% | 6.38 mm | 1.74° | False |
| 组合接触 + 只按指尖停止 | 33.9% | 21.38 mm | 9.88° | False |
| 几何接触 + 关闭伺服/闭合辅助 | 12.1% | 40.45 mm | 37.90° | False |

原方案接触窗口在帧 525、537、618、637 因原始任务接触变为 False、拇指接触消失而重置。五指有几次同时接触，但仍无法达到严格稳定保持。因此**不能**把接触比例等同于物体静止或抓握成功。

## 2. 四种可运行的控制策略

仅在 `--task wrap --mapping-method legacy --sim mujoco` 的**非六维耦合伺服**路径中启用。其他任务的控制行为不变。

| 参数 `--wrap-servo-policy` | 实际控制差别 | 检验假设 |
| --- | --- | --- |
| `original`（默认） | v1.3.3 原始伺服；增益 0.80、`stop_on_contact: any` | 原基线 |
| `low_gain` | 只将 WRAP 伺服增益降至 0.40；其他逻辑保持 | 降低伺服推进强度是否有益 |
| `thumb_recovery` | 保留低增益，并在拇指**曾有真实受力接触**、随后失接触且其他指仍接触时，执行有限持续时间、有限关节步长、限制目标追逐速度的拇指恢复 | 能否减少拇指失接触、改善连续稳定 |
| `coordinated` | 在 `thumb_recovery` 基础上，恢复期间让已接触的其余手指维持当前关节位置；未接触手指仅可做极小步长动作 | 约束单侧推进是否减少物体漂移 |

**公平性设计：** 后两组都与 `low_gain` 共享 0.40 基础增益。比较 `thumb_recovery - low_gain` 才能单独衡量拇指恢复的新增贡献；比较 `coordinated - thumb_recovery` 才能单独衡量其他四指协调的新增贡献。不得把从 `original` 到 `coordinated` 的全部差异都归功于协调机制。

**控制安全边界（仅仿真候选值）：**

- 拇指恢复基础增益 0.22；每次关节恢复步长不超过 0.8°；一次丢失最多恢复 18 控制帧。
- 恢复目标每帧最多跟随移动物体 0.35 mm；相对最后一次实接触目标的总跟随不超过 3 mm。
- 协调策略在拇指恢复期间保持已接触其他手指的位置；未接触其他手指单帧修正不超过 0.2°。
- `src/main.py` 的低通滤波、关节限位、最终速度限幅仍然保留；**不得**把这些仿真参数直接用于实际硬件。

> 所有策略的成功率始终依据 MuJoCo *本帧有效接触力* 与严格相对位姿窗口计算。时序滤波只用于控制反馈，不能因为滤波延长而计为成功。

## 3. Windows PowerShell 一键运行

在项目根目录并已激活 `.venv` 时，先跑一组调试试次（4 个条件 × trial 0）：

```powershell
python experiments/benchmark_wrap_servo_policy.py --trials 0 --steps 800
```

通过后再跑 8 个不同合成数据试次（4 × 8 = 32 次仿真）：

```powershell
python experiments/benchmark_wrap_servo_policy.py --trials 0,1,2,3,4,5,6,7 --steps 800 --output-dir outputs/wrap_servo_policy_8trials
```

如果希望排除仿真过程中**越界自动复位**导致的数据污染，四种方法均使用同一个开关，另存一批独立对照，不要与默认实验混合：

```powershell
python experiments/benchmark_wrap_servo_policy.py --trials 0 --steps 800 --no-auto-reset-object --no-inactive-object-reset --output-dir outputs/wrap_servo_no_teleport
```

上述两个开关分别禁用**工作空间越界**触发的自动复位、以及**非活动阶段**回到原位；仍保留任务刚开始时的一次初始复位以保证公平起点。`object_reset_this_step` 会标明仿真帧的复位事件。与先前默认设置的结果不能混为一批统计。

如果希望直观看圆柱与手指：

```powershell
python -m src.main --sim mujoco --render --realtime --input dataset --dataset-trial 0 --task wrap --steps 800 --camera closeup --contact-detector geometry_only --wrap-servo-policy coordinated --output-dir outputs/wrap_visual --run-name wrap_trial0_coordinated
```

只读取已有结果、重建汇总表（不运行仿真）：

```powershell
python experiments/benchmark_wrap_servo_policy.py --trials 0 --summarize-only
```

当前环境和原项目都不要求 Python 3.11；你已配置好的 Python 3.12 虚拟环境可沿用。**不需要重新 `git init`。**

## 4. 结果文件如何看

每次运行产生 `wrap_trial0_original.csv` / `_summary.json` 等独立文件，批量脚本随后生成：

- `per_trial.csv`：每个试次、每种方法的物理接触率、滚动稳定帧比例、结束时稳定成功、历史最长连续稳定帧数、最大相对位姿跨度、物体相对初始位置最大位移、平均伺服误差、控制状态切换数、拇指掉接触次数、恢复帧数、协调帧数、自动复位次数，以及平均手指作用在物体上的合力 XY 大小。
- `aggregate.csv`：按方法汇总均值、标准差，以及相同 trial 与 `original` 的成对差值（仅当包含 `original` 时）。
- `report.md`：自动生成的简短结果对比表（全部来自真实运行输出，不预填性能）。
- `manifest.json`：数据文件 SHA-256、配置 SHA-256、试次、接触判定器、是否允许越界自动复位以及比较限制。

**新增 CSV 物理诊断字段：** `raw_thumb_contact` 等五指受力接触标志、`thumb_normal_force_n` 等五指法向力、`thumb_force_on_object_x_n/y_n/z_n` 等各指对物体的**世界坐标系三维接触力**、`object_pos_x_m/y_m/z_m`、`object_reset_this_step`，以及 `servo_thumb_recovery_active` / `servo_wrap_coordination_active` / `servo_thumb_loss_event`。力采用 MuJoCo `mj_contactForce` 和 `contact.frame` 转换，并按接触对中物体位于 geom1 或 geom2 处理符号。注意**合力不包含物体与桌面的接触力**，因此不能用五指合力单独推断物体加速度。

## 5. 判断「真正有效」的最低条件

1. 不仅比较 `contact_ratio`，必须优先比较严格 `stable_success_proxy`、最长连续稳定帧数和物体相对手掌的平移/旋转跨度。
2. 检查是否出现“接触切换少，但一直未接触”的伪改善；同时对比 `raw_thumb_contact` 失联次数和接触力向量。
3. 不要把项目中 `stable_contact_proxy` 的历史简化判据与新的 `stable_success_proxy` 混用；后者采用自由物体、1 秒滚动窗口、相对手掌平移 1 mm/旋转 1° 阈值。
4. 跨试次使用成对比较。8 个试次来自**同一合成数据集**，不是 8 名独立受试者，更不能据此发表真实硬件成功率。
5. 只有在多个场景尺寸、初始位置、摩擦、质量、随机扰动及独立手套数据上均观察到稳定收益，才能进一步讨论稳健性和创新贡献。若新策略未改善，就如实报告失败。

## 6. 已测试与未测试

- 在当前执行环境中：策略逻辑、拇指掉接触触发、速率/目标限幅、超时清除、单元诊断和批量结果解析均通过单元测试。
- **尚未在当前执行环境中运行真实 MuJoCo 物理对照**：此环境未安装 `mujoco` 且无法联网获取。必须在已经可运行 MuJoCo 的 Windows `.venv` 中执行上面的批量命令，然后以实际 CSV/JSON 填写实验结果。
- 研究意义来自可复现实验设计与后续验证，而不是给旧技术组件换名称；本候选版不主张方法已经具有可发表的新颖性。
