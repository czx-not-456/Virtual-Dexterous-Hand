# v0.3 文献驱动设计说明

本文件说明 Virtual-Dexterous-Hand-v0.3 如何把项目参考文献中与“结构、映射、意图、虚拟评测”直接相关的信息落实到工程中。

> 说明：v0.3 是**文献驱动的工程验证模型**，不是 CasiaHand 官方数字孪生。没有来源支持的几何尺寸、关节活动范围、腱轮半径、摩擦参数等均保持为项目可调近似值，不冒充论文实测参数。

## 1. CasiaHand 结构信息 → 15 DoF / 7 actuator 控制架构

来源：Yan D, Wang P, Zhang T, et al. *CasiaHand: Design and Evaluation of a 15-DoF Tendon-Driven Anthropomorphic Robotic Hand*, IEEE RA-L, 2025.

文献明确描述：

- CasiaHand 共 15 DoF、7 个执行器；
- 拇指由 3 个独立关节全驱动；
- 四根非拇指各由单根腱控制 MCP/PIP/DIP 三个关节；
- 非拇指关节使用扭簧恢复并提供被动顺应；
- 非拇指 MCP/PIP/DIP 扭簧刚度分别为 24.106 / 12.857 / 12.857 mNm/rad。

v0.3 对应实现：

- `configs/robot_hand.yaml`：显式声明 `nominal_dof: 15`、`effective_actuators: 7`；
- `RobotHandModel.actuator_targets_from_joint_targets()`：把 15 维姿态投影到 7 维控制空间；
- `humanoid_hand_v03.xml`：拇指 3 个 position actuator + 4 根 fixed tendon actuator；
- 非拇指 12 个 hinge joint 写入论文给出的扭簧刚度；
- 掌面、鱼际和指尖加入真实碰撞面，支持后续接触评估。

工程近似：论文没有在当前材料中给出完整腱轮半径和每个关节的精确传动系数，因此 v0.3 的 tendon coefficient 仍使用项目可调协同系数，不声称为 CasiaHand 官方传动参数。

## 2. 混合关节-笛卡尔映射 → “位置 + 相对向量 + 关节拓扑 + 平滑 + 安全”

来源：Huang Y, Wang Z, Shen X, et al. *Human-Like Dexterous Manipulation for the Anthropomorphic Hand-Arm Robotic System via Teleoperation*, ICIRA 2023.

文献将人手到机器人手映射构造成非线性优化问题，并同时考虑：

- 关键点/向量关系；
- 对应关节角相似性；
- 时间连续性；
- 精细捏合中的指尖接近；
- 非目标手指之间的安全距离与自碰撞。

v0.3 对应实现：

- `src/retargeting/constraints.py::vector_shape_error()`：关键指尖相对向量误差；
- `fingertip_collision_penalty()`：允许 PINCH 的 thumb-index 接近，同时抑制其他异常接近；
- `IntentDrivenRetargeter` 目标函数扩展为：

  `E = w_pos E_pos + w_vector E_vector + w_topo E_topo + w_smooth E_smooth + w_coupling E_coupling + w_collision E_collision`

- 仍采用 SLSQP，保持与项目申报路线一致；
- 权重继续由 `NEUTRAL/PINCH/WRAP` 意图动态切换。

工程近似：文献中的权重和距离阈值针对其 MANO/CASIA Hand 系统；v0.3 没有直接照搬数值，而是保留为项目配置参数，避免把不同尺寸模型的参数错误迁移。

## 3. 运动序列意图 → 短时序鲁棒识别

来源：Huang Y, Fan D, Yan D, et al. *Human-Robot Collaborative Tele-Grasping in Clutter With Five-Fingered Robotic Hands*, IEEE RA-L, 2025.

文献使用人体运动序列进行实时意图识别，并强调连续、平滑的抓取过程。

v0.3 尚未具备论文训练数据，因此没有伪造/复现 Bi-GRU；而是：

- 在 `IntentRecognizer` 中加入 `sequence_window_frames`；
- 对 `pinch_distance` 和 `wrap_score` 做短时序平均；
- 再结合原有迟滞阈值与 dwell frame，减少单帧噪声引发的误触发。

这一步是“受文献启发的轻量实现”，后续获得真实数据手套数据集后，可以将其替换为训练型序列意图模型。

## 4. 欠驱动手评测 → 接触区 + 姿态稳定性 + 指尖接触率

CasiaHand 论文提出分层评测，手-物交互部分特别关注：

- spatial stability；
- sectional contact（thumb / fingers / palm）；
- 抓取过程中物体是否滑移；
- 柱状物体的包络抓握以及掌部参与。

2025 tele-grasping 论文进一步使用 contact ratio 评价手-物接触改善，并用 success rate / completion time / NASA-TLX 评价真实遥操作任务。

v0.3 新增 MuJoCo 在线指标：

- `contact_count`；
- `contact_sections`；
- `thumb_contact`；
- `finger_contact`；
- `palm_contact`；
- `tip_contact_ratio`；
- `contact_streak_frames`；
- `orientation_drift_deg`；
- `object_displacement_m`；
- `stable_contact_proxy`。

其中姿态稳定阈值默认 3°，与 CasiaHand 论文的 spatial stability 判据一致。

注意：`stable_contact_proxy` 只是仿真静态/局部接触代理指标，不等同于论文中“抓起并移动物体”的正式任务成功率。

## 5. 测试物体设计 → 包络 / 捏合 / 形状适应三类场景

CasiaHand 的评测对象包括圆柱、薄物体、球/不规则物体等，以覆盖：

- 包络抓握；
- 指尖精细操作；
- 形状适应。

v0.3 通过 `--task` 选择：

- `wrap`：圆柱；
- `pinch`：薄块；
- `sphere`：球体。

使用同一 MJCF，在加载前动态修改 task object 的形状、尺寸与初始位置。

## 6. 暂未直接实现、但已留出接口的文献内容

以下内容暂不应在 v0.3 中宣称“已完成”：

- 真实 CasiaHand 官方 CAD/MJCF/URDF 数字孪生；
- 真实数据手套 + 三 IMU 的人体手臂联合捕捉；
- 论文中的 Bi-GRU 意图网络；
- CVAE 连续抓取生成；
- 点云场景感知和 Mask R-CNN；
- 梯度式 mesh-level grasp refinement；
- 真实力/触觉反馈；
- NASA-TLX 用户实验。

这些可作为 v0.4 以后、真实硬件与数据到位后的扩展方向。
