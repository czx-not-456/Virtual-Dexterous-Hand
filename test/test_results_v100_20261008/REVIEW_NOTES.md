# 项目审查记录

审查日期：2026-10-08。范围为当前工作区源码、配置、测试、输入数据、版本记录与说明文档。

## 当前实现依据

| 项目内容 | 核实位置 | 结论 |
| --- | --- | --- |
| 主程序与输入切换 | `src/main.py`、`src/glove/adapter.py` | `sim` 支持 dataset/mock；`glove` 使用实采标定样本回放与 base mapping |
| 真实设备通信 | `src/glove/real_driver_template.py` | 构造函数直接抛出 NotImplementedError，设备实时采集尚未实现 |
| 人手特征 | `src/hand_model/`、`src/features/` | 15 个语义关节，指尖运动学与协同特征；意图枚举只有 NEUTRAL/PINCH/WRAP |
| CH-M6 模型 | `CH-M6/CH-M6_L.xml`、`configs/robot_hand.yaml` | 11 DoF、11 个位置执行器；人手通道通过配置映射为模型关节 |
| 模型适配 | `src/simulation/ch_m6_adapter.py` | 原模型和 STL 只读，运行时补充接触代理、指尖 site 和任务场景 |
| 标准任务 | `configs/simulation.yaml` | 当前只配置 neutral、pinch、wrap；历史六任务不属于当前默认目录 |
| 任务评估 | `src/evaluation/task_evaluator.py`、`src/simulation/mujoco_env.py` | 连续 12 个活动帧；PINCH 要求 contact，WRAP 要求 stable；当前姿态稳定阈值 5° |
| 个性化标定 | `experiments/compare_glove_calibration.py` | 每姿态第 4 帧留出，合计 6 个样本、6 个目标输出，读取 manifest 预测 |
| 实采任务对照 | `experiments/compare_glove_tasks.py` | 每组 240 帧，其中 30 帧张开、210 帧任务姿态；同一留出姿态重复回放 |
| Unity 接口 | `src/simulation/unity_bridge.py` | 仅有发送 joint_positions/intent 的 UDP 桥，未发现 Unity 工程与接收端 |

## 文档与证据的限制

- 原有同名简报属于旧模型阶段，已备份为本目录的 `项目进度简报_20260921_原稿.md`。新简报沿用用户指定文件名，明确采用 2026-10-08 的审查和测试事实，不将当前结果写成 9 月 21 日已完成。
- `PROJECT_BASIS.md`、`CHANGELOG_v0.md`、`MIGRATION.md` 和旧测试目录包含历史架构和数值；当前结论应优先依据运行代码、配置与本次实测。
- README 的“WRAP vector MSE 约退化 8.47% / coupling 改善”来自旧阶段。当前 `synergies: {}`，耦合惩罚为零，不能用该历史论述证明当前 CH-M6 耦合改善。
- 实采交付只有 xlsx、base mapping 和 manifest，未发现对应 `.pt` 权重。manifest 虽记载外部训练与模型路径，但本项目尚不能执行该个性化模型的连续在线推理。
- 实采标定集为 6 姿态 × 4 帧，不能支持多人泛化结论；重复仿真帧不能视为独立受试样本。
- 固定手掌、工程接触代理和 PINCH 接近阶段夹具限制了结果含义；接触成功与稳定成功都不等价于抓起、搬运、放置成功。
- 本次无界面运行验证物理和数据流程，未人工检验 Viewer 鼠标交互、真实手套设备、Unity 接收端或硬件操作。
- 本次未重新核验申报书中的原定月度节点，不沿用旧简报中“节点滞后”的断言。
- `v100` 是用户指定的测试归档名称；当前代码没有对应已确认的 v1.0.0 发布声明。

## 后续工作建议

1. 补齐真实手套协议、SDK、时间戳和同步规则，完成实时采集到 CH-M6 的闭环。
2. 获取个性化模型权重、预处理定义与推理接口，验证连续动作而非固定姿态回放。
3. 确认实际 CH-M6 驱动数、distal 关节关系、指腹参数、摩擦和任务夹具，完成模型标定。
4. 按任务分别改善自由物体稳定性与处理延迟，加入抬升/搬运评价阶段。
5. 明确 Unity、标准物体、受试者与重复试验的验收要求，整理论文和结题材料。
