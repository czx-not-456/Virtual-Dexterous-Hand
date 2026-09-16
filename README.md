# v0.4.0：模拟数据集 + 标准物体评测 + 自动结果汇总

> **v0.4.5 PINCH note:** the calibration block is 90×30×132 mm and is temporarily held upright during the PINCH approach. It is released immediately after bilateral thumb/index fingertip contact is observed, so subsequent stability is evaluated with the object free.

> **v0.4.2 combined release:** this package includes both the validated WRAP contact tuning and the PINCH fingertip-contact fix. See `CHANGELOG_v0.4.2.md`.

本版本继续采用**模拟数据**，不依赖真实数据手套。核心目标是把现有“能做捏合/抓握”的演示工程推进为可重复、可量化、可批量验证的大创实验系统。

新增内容：

- `datasets/synthetic_glove_v1.csv`：8 组可复现模拟数据试验，每组 600 帧；
- `CSVGloveDatasetDriver`：像真实手套一样逐帧回放 CSV；
- 标准物体任务：`pinch / wrap / sphere / card / bottle / box`；
- PINCH 可达性优化：任务空间目标不再强制指尖对准盒体中心线，并增加拇指/食指预定位先验；
- 通用 box / cylinder / sphere 表面目标生成；
- 任务级量化评估：接触率、稳定接触率、首次接触时间、连续稳定帧数、P95 延迟、关节 RMSE、指尖接触率、物体姿态漂移等；
- 每次运行自动生成 `CSV + summary.json`；
- `experiments/benchmark_synthetic.py`：批量跑多个物体、多组模拟数据，并生成 CSV、Markdown 报告和 HTML 看板。

> 当前模型手掌基座固定，因此 `success_proxy` 定义为“连续稳定接触代理”，**不等同于抓起并搬运成功率**。若后续加入手腕/机械臂自由度，再增加 lift / transport 指标。

## 推荐运行流程

安装依赖：

```powershell
cd D:\Vhand\Virtual-Dexterous-Hand-main
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

先跑自动测试：

```powershell
pytest -rA
```

用第 0 组模拟数据跑 PINCH：

```powershell
python -m src.main --sim mujoco --render --realtime --input dataset --dataset datasets\synthetic_glove_v1.csv --dataset-trial 0 --task pinch --steps 600 --camera closeup
```

跑 WRAP：

```powershell
python -m src.main --sim mujoco --render --realtime --input dataset --dataset datasets\synthetic_glove_v1.csv --dataset-trial 0 --task wrap --steps 600
```

其他物体：

```powershell
python -m src.main --sim mujoco --render --realtime --task sphere --steps 600
python -m src.main --sim mujoco --render --realtime --task card --steps 600 --camera closeup
python -m src.main --sim mujoco --render --realtime --task bottle --steps 600
python -m src.main --sim mujoco --render --realtime --task box --steps 600
```

每次运行后 `outputs/` 会出现：

```text
run_YYYYMMDD_HHMMSS_<task>.csv
run_YYYYMMDD_HHMMSS_<task>_summary.json
```

批量评测 6 类物体 × 5 组模拟数据：

```powershell
python experiments\benchmark_synthetic.py --trials 5 --steps 600
```

输出目录：

```text
outputs/benchmark/
├── benchmark_runs.csv
├── benchmark_summary.csv
├── BENCHMARK_REPORT.md
└── benchmark_dashboard.html
```

重新生成模拟数据集：

```powershell
python scripts\generate_synthetic_dataset.py --trials 8 --frames 600 --seed 42
```

---


## v0.3.4：基于物体位姿的任务空间接触伺服

在 v0.3.3 的真实运行数据中，PINCH 仍为 0% 接触，WRAP 虽出现少量接触但圆柱会在正式 WRAP 前被推倒。v0.3.4 因此新增：

- Object-aware task-space servo：使用 MuJoCo 指尖 Jacobian 把目标指尖导向当前物体表面；
- PINCH：拇指 3 DoF + 食指单腱有效方向闭环；
- WRAP：拇指与四根非拇指分别朝圆柱当前表面收敛；
- Task gating：非目标意图期间保持物体标准初态，进入目标意图时复位后释放；
- CSV 记录任务空间误差，便于下一轮基于数据调参。

详见 `CHANGELOG_v0.3.4.md` 与 `CONTACT_VALIDATION_v0.3.4.md`。


## v0.3.2 接触闭环辅助

v0.3.2 新增任务级接触闭环辅助：PINCH 时仅对拇指/食指追加小幅闭合，WRAP 时对五指分别追加闭合；某根手指与任务物体接触后会停止继续增加该手指的额外闭合量。接触统计现在区分手-物接触与物体-环境接触，并记录逐指/逐指腹接触状态。详见 `CONTACT_VALIDATION.md`。

# 基于数据手套的手部动作采集与虚拟灵巧手交互系统（v0.3）

v0.3 在 v0.2 已经跑通的 **Mock 数据手套 → 姿态解算 → 意图识别 → 动态映射 → MuJoCo** 链路上，进一步把项目参考文献中与结构、映射和评测直接相关的信息落实到工程中。

完整链路：

> Mock 数据手套 → 自适应标定 → 滤波/归一化 → 姿态解算 → 短时序意图识别 → 文献驱动混合映射 → 15 DoF / 7 actuator 欠驱动 MuJoCo 手 → 手-物接触与稳定性评测 → CSV 记录

> **重要定位**：v0.3 是“文献驱动的工程验证模型”，不是 CasiaHand 官方数字孪生。没有可靠来源支持的 CAD 尺寸、腱轮半径、摩擦参数、真实关节限位等仍采用项目可调近似值。

详细依据见：

`LITERATURE_INFORMED_DESIGN.md`

---

## 1. v0.3 相比 v0.2 的核心升级

### 1.1 由 15 个独立驱动改为 15 DoF / 7 actuator

参考 CasiaHand 2025 论文的结构：

- 拇指 3 个关节独立驱动；
- 食指/中指/无名指/小指分别由单根 flexor tendon 驱动 3 个关节；
- 总计 15 DoF、7 个有效执行器。

对应文件：

- `configs/robot_hand.yaml`
- `src/hand_model/robot_hand.py`
- `models/dexterous_hand/humanoid_hand_v03.xml`

运行时会显示：

```text
Robot architecture: nominal DoF=15, effective actuators=7
```

### 1.2 非拇指加入论文给出的扭簧刚度

v0.3 MJCF 为非拇指写入：

```text
MCP: 0.024106 N·m/rad
PIP: 0.012857 N·m/rad
DIP: 0.012857 N·m/rad
```

这对应 CasiaHand 论文中的 24.106 / 12.857 / 12.857 mNm/rad。

其作用不是把三个关节锁死，而是让单腱驱动下的手指具有一定被动顺应趋势。

### 1.3 映射算法从“位置+拓扑”扩展为混合目标

v0.3 目标函数包含：

```text
E_pos       关键末端位置
E_vector    关键指尖相对向量/手形关系
E_topo      拓扑/关节相似
E_smooth    时间平滑
E_coupling  欠驱协同软约束
E_collision 简化自碰撞安全
```

仍然使用 SLSQP，并由 `NEUTRAL / PINCH / WRAP` 动态调权。

这比 v0.2 更接近参考文献中的“关节 + 笛卡尔/关键点 + 平滑 + 安全约束”混合映射思想。

### 1.4 意图识别加入短时序信息

2025 tele-grasping 工作强调使用人体运动序列识别抓取意图。v0.3 当前没有真实训练数据，因此**没有伪造 Bi-GRU 模型**，而是加入：

- `sequence_window_frames`；
- pinch distance 短时序均值；
- wrap score 短时序均值；
- 原有迟滞阈值；
- dwell frame。

这能减少单帧噪声误触发，同时保留以后替换成训练型时序模型的接口。

### 1.5 MuJoCo 加入真实可统计的接触区

v0.2 的掌面主要用于视觉展示；v0.3 将以下区域真正加入碰撞：

- thumb/thenar；
- fingers；
- palm；
- 五个 fingertip pads。

因此现在不仅能“看动作”，还能记录手-物接触。

### 1.6 新增文献对应的稳定性/接触指标

每帧 CSV 新增：

```text
contact_count
contact_sections
thumb_contact
finger_contact
palm_contact
tip_contact_ratio
contact_streak_frames
orientation_drift_deg
stable_orientation
stable_contact_proxy
object_displacement_m
```

其中 `orientation_drift_deg` 默认采用 **3°** 作为稳定性阈值，与 CasiaHand 论文中的 spatial stability 判据一致。

`stable_contact_proxy` 只是本项目仿真的代理指标，不等同于正式“抓起并搬运成功率”。

### 1.7 三类测试物体

v0.3 支持：

```text
--task wrap    圆柱，观察包络抓握
--task pinch   薄块，观察拇指-食指精细捏合
--task sphere  球体，观察形状适应
```

同一 MJCF 会在加载前由 Python 动态修改测试物体，不需要维护多份场景。

---

## 2. 安装环境

Windows 10/11，推荐 Python 3.10–3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. 先跑测试

```powershell
pytest -rA
```

生成环境结果：

```text
15 passed, 1 skipped
```

跳过的是 MuJoCo runtime schema 测试，因为生成环境没有安装 MuJoCo。

你的电脑已经能打开 MuJoCo，因此本机正常应当进一步执行该测试；若全部成功，理论上会看到：

```text
16 passed
```

---

## 4. 纯算法运行

```powershell
python -m src.main --sim none --steps 900
```

Mock 手势仍循环：

```text
OPEN → NEUTRAL → PINCH → NEUTRAL → WRAP → OPEN
```

终端会显示意图切换以及 15 DoF / 7 actuator 架构信息。

---

## 5. MuJoCo 推荐运行方式

### 5.1 包络抓握场景

```powershell
python -m src.main --sim mujoco --render --steps 0 --realtime --task wrap
```

重点观察：

- 四根非拇指在单 flexor tendon + spring 下的协同弯曲；
- thumb/fingers/palm 是否逐渐形成多区域接触；
- 圆柱是否发生明显姿态滑移。

### 5.2 精细捏合场景

```powershell
python -m src.main --sim mujoco --render --steps 0 --realtime --task pinch --camera closeup
```

重点观察：

- 拇指与食指是否对薄块形成接触；
- 其余三指是否保持相对开放；
- `tip_contact_ratio` 和接触连续性是否变化。

### 5.3 球体适应场景

```powershell
python -m src.main --sim mujoco --render --steps 0 --realtime --task sphere
```

用于观察欠驱手指面对曲面时的接触适应趋势。

### 5.4 自由视角

```powershell
python -m src.main --sim mujoco --render --steps 0 --realtime --task wrap --camera free
```

---

## 6. 映射对照实验

```powershell
python experiments/compare_mapping.py
```

生成环境当前 Mock 参数下：

```text
PINCH position MSE improvement:        9.58%
PINCH vector MSE improvement:         40.03%
WRAP coupling penalty improvement:    83.64%
WRAP vector MSE improvement:         -41.77%
```

WRAP 的 vector MSE 变差并不意味着算法失效：WRAP 模式显式提高了欠驱协同目标的权重，因此会牺牲部分“与人手相对向量完全一致”的目标。这正是多目标、意图驱动映射要研究的权衡。

这些都是 Mock 工程验证结果，**不能直接作为论文最终实验数据**。

---

## 7. 分析运行 CSV

每次 `src.main` 会生成：

```text
outputs/run_YYYYMMDD_HHMMSS.csv
```

分析：

```powershell
python experiments/analyze_run.py outputs\run_你的时间戳.csv
```

可快速查看：

- 平均端到端延迟；
- 平均优化耗时；
- 最大物体姿态漂移；
- 平均指尖接触率；
- 接触帧数；
- stable-contact proxy 帧数；
- 各意图持续帧数。

---

## 8. 当前仍未完成的内容

v0.3 **没有**宣称实现以下内容：

- 实验室真实数据手套 SDK；
- CasiaHand 官方 CAD/MJCF/URDF；
- 精确腱轮半径与腱路几何；
- 真实力传感器/触觉反馈；
- Bi-GRU 意图识别；
- CVAE 连续抓取生成；
- 点云场景感知；
- mesh-level grasp refinement；
- NASA-TLX 用户实验。

下一阶段最优先的是：

1. 接入真实数据手套；
2. 获取实验室目标灵巧手官方模型/机械参数；
3. 用真实数据重新标定意图阈值与映射权重；
4. 在 YCB/标准物体上正式进行精度、延迟、稳定性和抓取任务实验。

---

## 9. 参考文献对应关系

本版直接吸收的信息主要来自：

1. Yan D, Wang P, Zhang T, et al. *CasiaHand: Design and Evaluation of a 15-DoF Tendon-Driven Anthropomorphic Robotic Hand*. IEEE Robotics and Automation Letters, 2025.
2. Huang Y, Wang Z, Shen X, et al. *Human-Like Dexterous Manipulation for the Anthropomorphic Hand-Arm Robotic System via Teleoperation*. ICIRA 2023.
3. Huang Y, Fan D, Yan D, et al. *Human-Robot Collaborative Tele-Grasping in Clutter With Five-Fingered Robotic Hands*. IEEE Robotics and Automation Letters, 2025.
4. Li Y, Wang P, Li R, et al. *A Survey of Multifingered Robotic Manipulation: Biological Results, Structural Evolvements, and Learning Methods*. Frontiers in Neurorobotics, 2022.

更详细的“文献结论 → 代码改动”对应表见 `LITERATURE_INFORMED_DESIGN.md`。

---

## v0.3.2 可视化修复

v0.3.2 解决了本机 MuJoCo 验证中暴露的两个问题：

1. 手模型视觉上像有“六根手指”：已移除细长 forearm 几何，仅保留五个 digit root，并以宽矩形 wrist base 表示腕部。
2. `overview / closeup / side` 视角无法自由拖动：现在这三个名称均为 **FREE camera 的初始视角预设**，启动后可继续用鼠标自由旋转、平移、缩放。

详见 `CHANGELOG_v0.3.2.md` 与 `CAMERA_CONTROLS.md`。

## v0.3.4：任务工作区校准

v0.3.4 根据真实运行 CSV 校正了 PINCH 薄块与 WRAP 圆柱的位置/尺寸，并调整接触辅助策略。详见 `CHANGELOG_v0.3.4.md` 与 `CONTACT_VALIDATION_v0.3.4.md`。
