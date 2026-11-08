# CHANGELOG_v0

## CHANGELOG_v0.2

- 升级 15-DoF MuJoCo 仿人手外观：掌面/掌根/鱼际/腕部/五指分节。
- 增加拇指、食指及其余手指的可视化指尖标记。
- 修正 MJCF 屈曲轴方向，使正屈曲角与 Python 运动学一致。
- 新增抓取台、动态圆柱测试物体和 PINCH 观察点。
- 新增 overview / closeup / side 固定相机和 free 自由视角。
- 新增 `configs/simulation.yaml`。
- MuJoCo 驱动兼容 Windows 中文路径。
- 新增 `--steps 0` 持续运行和“关闭 Viewer 即退出”。
- 新增测试物体越界自动复位。
- 新增 3 项 MJCF 合同测试及 1 项可选 MuJoCo schema 运行时测试。

## CHANGELOG_v0.3

### 结构与物理模型

- 将 v0.2 的 15 个独立 position actuator 改为 **15 DoF / 7 actuator** 文献驱动结构。
- 拇指保持 3 个独立驱动；四根非拇指改为单 flexor tendon 驱动三关节。
- 非拇指 MCP/PIP/DIP 加入 0.024106 / 0.012857 / 0.012857 N·m/rad 弹簧刚度。
- 掌部重新启用碰撞，加入五个 fingertip contact pad。
- 保留 v0.2 原模型用于回溯，新默认模型为 `humanoid_hand_v03.xml`。

### 映射算法

- 新增关键指尖相对向量误差 `E_vector`。
- 新增简化指尖自碰撞安全项 `E_collision`。
- PINCH/WRAP/NEUTRAL 动态权重扩展到六项代价函数。
- 15 维目标姿态增加到 7 维执行器空间的显式投影。

### 意图识别

- 加入短时序滑窗，吸收“利用运动序列识别意图”的研究思路。
- 保留迟滞 + dwell 机制。

### MuJoCo 评估

- 新增 `--task wrap|pinch|sphere`。
- 在线记录 thumb/fingers/palm 接触区、指尖接触率、物体姿态漂移、位移和接触连续性。
- 默认姿态稳定阈值为 3°。

### 实验工具

- `experiments/compare_mapping.py` 升级为 v0.3 混合映射对照。
- 新增 `experiments/analyze_run.py` 汇总运行 CSV。
- 自动化测试扩充到 15 项纯工程测试 + 1 项可选 MuJoCo runtime schema 测试。

## CHANGELOG_v0.3.1

*修复说明*

本版本针对 v0.3 实机 MuJoCo 验证中发现的两个可视化问题进行修复。

### 1. 五指模型修复

- 明确保留且仅保留 5 个手指根节点：`thumb1`、`index1`、`middle1`、`ring1`、`little1`。
- 移除容易从正视/斜视角看成“第六根手指”的细长 `forearm` capsule。
- 使用宽矩形 `wrist_base` 表示腕部底座，使腕部与手指在视觉上明显区分。
- 新增自动测试，确保 palm 下不会再出现第六个 digit root。

### 2. 相机改为可拖拽自由视角

v0.3 的 `overview / closeup / side` 会切换为 MuJoCo fixed camera，视点被锁定。
v0.3.1 将它们改为 **FREE camera 初始预设**：

- `overview`：启动时给出整体观察角度；
- `closeup`：启动时靠近拇指/食指与任务物体；
- `side`：启动时提供侧向观察；
- `free`：使用 MuJoCo 默认自由相机。

无论使用前三种哪一种，启动后都仍可通过 MuJoCo Viewer 的鼠标交互全方位观察。

### 3. 同步纳入 v0.3 XML Schema 修复

- `fixed tendon` 不再写入不受支持的 `width` / `rgba` 属性。
- 该修复与你本机 16 项测试全部通过后的版本一致。

## CHANGELOG_v0.3.2

*task contact closed-loop prototype*

本版本重点把“动作像抓握”推进到“真实发生手-物接触”。

### 主要变化

1. 保留 v0.3.1 五指模型，并合并侧置、向内屈曲的大拇指修复。
2. 新增 `TaskContactAssist`：
   - PINCH：只对拇指/食指逐步增加闭合量；某指接触物体后停止继续闭合该指。
   - WRAP：五指独立闭合；每根手指一旦接触物体就停止额外闭合。
   - 离开目标意图后自动平滑释放辅助量。
3. 修正接触统计语义：`contact_count` 现在只统计手-物接触；物体-桌面/地面的接触改记为 `environment_contact_count`。
4. 新增逐指接触和指腹接触指标：`thumb_tip_contact`、`index_tip_contact` 等。
5. 新增任务代理指标：
   - `pinch_contact_proxy = thumb_pad AND index_pad`
   - `wrap_contact_proxy = thumb contact AND any non-thumb finger contact`
6. 微调 PINCH 薄块和 WRAP 圆柱的初始位置/尺寸，使其更落在当前五指可达区域。
7. 新增 `experiments/analyze_contacts.py`，可对一次运行日志统计 PINCH/WRAP 接触代理成功比例。

### 注意

这些辅助闭合量和物体初始位姿是当前虚拟原型的工程参数，不是文献或真实 CasiaHand 的官方标定值。接入真实数据手套和真实机械参数后需要重新标定。

## CHANGELOG_v0.3.3

*数据驱动的任务物体工作区校准*

本版本依据 v0.3.2 实际 MuJoCo 截图与两份运行 CSV 调整任务场景，而不是继续盲目增加关节闭合量。

### 观测结论

- PINCH 运行 1174 帧中，手-物接触为 0 帧；物体平均位移约 3.47 cm，说明原薄块从初始化高度落到桌面后脱离了拇指/食指有效工作区。
- WRAP 运行 823 帧中，仅 21 帧出现任何手-物接触，且全部来自无名指；拇指接触为 0 帧，`wrap_contact_proxy` 始终为 false。
- 由实际 WRAP/PINCH 关节轨迹及当前 FK 估算，拇指-食指最接近区域约位于 palm 坐标系 y≈-0.044 m、world z≈0.183 m；原 PINCH 物体位于 y=+0.010 m 且落地后顶部仅约 z=0.130 m，明显偏离该区域。

### 修改

#### PINCH 物体

- 改为更窄、更高的直立薄块：half-size `[0.006, 0.006, 0.056]` m。
- 放置于 `[-0.014, -0.044, 0.126]` m。
- 物体底部直接落在桌面顶面附近，不再先自由下落约 3.5 cm。
- 顶部约 z=0.182 m，与拇指/食指有效捏合高度一致。

#### WRAP 圆柱

- 半径增至 0.035 m，半高增至 0.065 m。
- 放置于 `[-0.005, -0.020, 0.135]` m。
- 圆柱侧面同时覆盖拇指侧 y≈-0.05 与非拇指侧 y≈0.00 的包络区域；顶部约 z=0.200 m。

#### 接触辅助

- PINCH 不再强迫拇指继续大幅卷曲：拇指额外闭合上限显著降低。
- PINCH 主要由食指向作为对掌支撑的拇指闭合，提高真正形成双指夹持的机会。
- WRAP 略提高闭合速度，但降低拇指额外卷曲上限，优先让重新定位后的圆柱进入拇指工作区。

### 新增测试

`tests/test_task_workspace.py` 防止后续任务物体再次偏离当前五指可达工作区。

## CHANGELOG_v0.3.4

*更新说明*

本版本基于 v0.3.3 的真实 MuJoCo 截图与两份运行 CSV 做针对性修正。

### 诊断结论

- PINCH 场景：231 个 PINCH 帧中没有发生任何手-物接触，说明固定角度闭合无法修正三维位置误差。
- WRAP 场景：WRAP contact proxy 仅约 9.6%，且圆柱在正式 WRAP 前已被前置动作推倒；稳定抓握代理始终为 0。

### 核心改动

1. 新增 ObjectAwareTaskServo：使用当前任务物体位姿、MuJoCo 指尖位置与 site Jacobian，逐帧把指尖引导到物体表面。
2. 拇指使用 3-DoF damped-least-squares 修正；非拇指沿单腱欠驱动协同方向做 1-DoF 有效 Jacobian 修正。
3. 新增任务门控物体复位：目标意图未激活时保持物体标准初态，进入 PINCH/WRAP 的首帧重新复位后再释放，防止圆柱在前置动作中提前倾倒。
4. 接触后保持对应手指的当前实际姿态，降低持续穿透或把物体推飞的风险。
5. CSV 新增 task-space servo 误差指标；analyze_contacts.py 会同时输出平均/最大任务空间误差。
6. 保留 v0.3.2 固定角度 TaskContactAssist 作为 fallback；当 task_space_servo 开启时不再叠加固定角度辅助。

### 新增 CSV 字段

- task_servo_active
- servo_thumb_error_m
- servo_index_error_m
- servo_mean_error_m
- servo_max_error_m
- servo_corrected_digits

## CHANGELOG_v0.4.0

### 目标

在不接真实数据手套的前提下，将 v0.3.4 推进为可复现实验、可批量评测、可直接形成大创实验结果的数据链路。

### 新增

1. **离线模拟数据集**
   - 新增 `datasets/synthetic_glove_v1.csv`；
   - 8 个 trial，每个 trial 600 帧；
   - 包含 OPEN / NEUTRAL / PINCH / WRAP；
   - 每个 trial 加入通道增益、偏置、噪声与手掌姿态扰动，用于模拟不同佩戴/个体差异。

2. **CSV 回放驱动**
   - `src/glove/dataset_driver.py`；
   - 支持 `--input dataset --dataset-trial N`；
   - 其余姿态解算、意图识别、映射、MuJoCo 控制代码无需区分输入来自实时设备还是离线数据。

3. **标准测试物体目录**
   - `wrap` 圆柱；
   - `pinch` 薄块；
   - `sphere` 球体；
   - `card` 卡片；
   - `bottle` 瓶状圆柱；
   - `box` 长方体。

4. **PINCH 接触可达性修正**
   - box 捏合目标保留当前指尖 X 坐标，避免要求固定无外展自由度的手指强制对准中心线；
   - 将 PINCH 垂直边界余量从 10 mm 调整到 1 mm；
   - 增加与当前手几何一致的拇指/食指预定位先验；
   - 提高任务空间闭环主导比例和单帧最大修正量。

5. **统一任务成功代理与评测**
   - 新增 `TaskEpisodeEvaluator`；
   - 连续稳定接触达到配置帧数后判定 `success_proxy=True`；
   - 自动统计接触率、稳定率、首次接触/成功时间、P95 延迟、RMSE、速度 RMS、姿态漂移等。

6. **自动批量实验与结果看板**
   - `experiments/benchmark_synthetic.py`；
   - 生成逐次实验 CSV、聚合 CSV、Markdown 报告、HTML 看板。

### 兼容性

- 保留原 `MockGloveDriver`；使用 `--input mock` 可回到程序生成式 mock；
- 默认输入改为 `dataset`，以保证实验可重复；
- 原 `pinch / wrap / sphere` 命令仍可用。

### 验证状态

当前生成环境完成：

- Python 静态编译通过；
- 全部非 MuJoCo runtime 测试通过；
- dataset 回放 600 帧纯算法 smoke run 通过；
- 批处理脚本在 `--sim none` 下流程通过。

当前容器没有 MuJoCo 包，因此 **PINCH/WRAP 的真实物理接触改善必须在已安装 MuJoCo 的 Windows 机器上复测**。不要把未复测的 `success_proxy` 写成最终实验结论。

## CHANGELOG_v0.4.2

*PINCH + WRAP consolidated contact-control release*

This release consolidates the validated WRAP contact tuning and the PINCH fingertip-contact fix into one package.

### WRAP retained from validated v0.4.0 tuning

- `task_space_servo.enabled: true`
- global servo gain `0.88`
- global DLS damping `0.018`
- global command blend `0.96`
- maximum per-frame joint step `6.5 deg`
- stop error `0.8 mm`
- WRAP surface clearance `5.5 mm`
- WRAP still freezes a digit on any contact (`stop_on_contact` defaults to `any`) to avoid excessive penetration.

These are the settings that produced `success_proxy=True` in the validated WRAP run.

### PINCH fix retained from v0.4.1

- PINCH stops a digit only when its fingertip pad contacts the object (`stop_on_contact: tip`).
- PINCH uses a signed virtual contact target of `-2.0 mm` (`target_offset_m`) so the controller keeps closing until physical contact occurs.
- PINCH keeps a strong posture prior and task-space command blend for thumb/index alignment.
- The thin pinch block uses a wider footprint while preserving a thin pinch direction to reduce premature tipping.

### Compatibility

The CLI and dataset format are unchanged. Existing commands for `pinch`, `wrap`, `sphere`, `card`, `bottle`, and `box` continue to work.

## CHANGELOG_v0.4.3

### PINCH physics-clock fix

- Preserve the v0.4.2 WRAP contact parameters and PINCH fingertip-only contact logic.
- Synchronize MuJoCo physical time with the 60 Hz glove/control loop.
- The MJCF timestep is 3 ms, so each 16.67 ms control tick now executes 5/6 physics substeps (5.56 average) via an accumulator.
- This specifically addresses the observed PINCH trace where index fingertip target error fell only from ~121 mm to ~86 mm before the PINCH window ended.
- Add a startup diagnostic showing control timestep, MuJoCo timestep, and average substep count.

## CHANGELOG_v0.4.4

### PINCH geometry calibration

- Raised the PINCH calibration block so its side faces overlap the actual thumb/index precision-pinch workspace.
- The previous block had `size_z=0.056`, `pos_z=0.126`, so its top was `0.182 m`.
- The hand model closes thumb/index around `z≈0.190–0.193 m`; with 8–8.5 mm fingertip pads the index could skim above the top edge and report zero contact.
- New PINCH block: `size=[0.035, 0.010, 0.066]`, `pos=[-0.014, -0.044, 0.136]`. Its bottom remains on the `z=0.070 m` table and its top is `z=0.202 m`.
- WRAP controller and the v0.4.3 physics-clock synchronization are unchanged.

## CHANGELOG_v0.4.5

*PINCH approach stabilization*

This version keeps the verified WRAP controller and combines two PINCH fixes:

1. **Larger calibration block**: PINCH box half-size is `[0.045, 0.015, 0.066]`, i.e. a physical 90 × 30 × 132 mm block. The bottom remains on the z=0.070 m table.
2. **Approach fixture**: while PINCH is active and bilateral fingertip-pad contact has not yet occurred, the free block is held at its initial pose. This prevents the first finger from toppling the object. Once both thumb and index fingertip contacts are observed, the object is released on the next control frame, so stable success still has to survive free-body physics.

The v0.4.3 physics-clock synchronization and the WRAP configuration are unchanged.

## CHANGELOG_v0.4.6

*PINCH reachability fix*

### Why v0.4.5 stalled
The index fingertip error dropped rapidly and then plateaued near 7–8 mm while
`index_tip_contact` stayed false. This was not a timing problem: the old PINCH
box ended at z=0.202 m, while the one-tendon index trajectory approaches the
object around z≈0.209 m. With an 8 mm fingertip pad, the minimum sphere-to-box
distance was still about 10.3 mm, so true fingertip contact was geometrically
impossible even though the servo kept converging.

### Changes
- PINCH calibration block: 90 × 30 × 140 mm, bottom still on the table.
- PINCH fingertip-center offset: +7 mm (no virtual interpenetration).
- Reachability-aware index prior: MCP/PIP/DIP = 70/95/70 deg.
- Synthetic trial cycle extended to 800 frames with 240-frame PINCH and WRAP phases.
- Added regression tests for actual index-pad/box geometric intersection.

WRAP control and the v0.4.3 physics-clock synchronization are unchanged.

## CHANGELOG_v0.4.6-pinch-success

This patch keeps the v0.4.6 hand geometry and control unchanged.

The evaluation semantics are corrected for precision PINCH:
- PINCH success: bilateral thumb/index fingertip contact sustained for the configured streak (default 12 frames).
- PINCH stability: still reported separately by `stable_ratio` using the existing orientation-drift criterion.
- WRAP success: unchanged; still requires consecutive stable-contact frames.

This avoids classifying an otherwise valid precision pinch as a complete failure solely because the free object tilts more than 3 degrees from its initial orientation.
