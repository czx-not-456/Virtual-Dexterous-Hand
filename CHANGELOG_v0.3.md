# CHANGELOG v0.3

## 结构与物理模型

- 将 v0.2 的 15 个独立 position actuator 改为 **15 DoF / 7 actuator** 文献驱动结构。
- 拇指保持 3 个独立驱动；四根非拇指改为单 flexor tendon 驱动三关节。
- 非拇指 MCP/PIP/DIP 加入 0.024106 / 0.012857 / 0.012857 N·m/rad 弹簧刚度。
- 掌部重新启用碰撞，加入五个 fingertip contact pad。
- 保留 v0.2 原模型用于回溯，新默认模型为 `humanoid_hand_v03.xml`。

## 映射算法

- 新增关键指尖相对向量误差 `E_vector`。
- 新增简化指尖自碰撞安全项 `E_collision`。
- PINCH/WRAP/NEUTRAL 动态权重扩展到六项代价函数。
- 15 维目标姿态增加到 7 维执行器空间的显式投影。

## 意图识别

- 加入短时序滑窗，吸收“利用运动序列识别意图”的研究思路。
- 保留迟滞 + dwell 机制。

## MuJoCo 评估

- 新增 `--task wrap|pinch|sphere`。
- 在线记录 thumb/fingers/palm 接触区、指尖接触率、物体姿态漂移、位移和接触连续性。
- 默认姿态稳定阈值为 3°。

## 实验工具

- `experiments/compare_mapping.py` 升级为 v0.3 混合映射对照。
- 新增 `experiments/analyze_run.py` 汇总运行 CSV。
- 自动化测试扩充到 15 项纯工程测试 + 1 项可选 MuJoCo runtime schema 测试。
