# CH-M6 稳定抓握评价 v2（完整工程，含 TQHG 在线映射 + Hold）

## 重要说明

- 本压缩包是在已提供的**原方案 E 完整工程**、**耦合伺服 Hold 消融补丁**、**TQHG 在线接入补丁**所能重建出的完整源码树基础上集成，不是直接读取你的 Windows 本机实时工程。
- 保留：`--mapping-method ridge6/pca4/fixed-blend/single-gate/tqhg`，原 TEQS 路径，以及 `--coupled-servo-ablation hold`。以前4维协同 `q_prev_command` 锚点修复已包含。
- **不要直接覆盖**原项目；建议解压到另一目录，验证通过后再选择迁移。
- 不包含 Windows `.venv`、你本机新建的 `CHM6_pinch_hold_diagnostic_only.xlsx`、本机 `outputs/compare_*` 及其日志；请从旧工程复制对应输入数据。原始项目自带的采集记录、模型、配置、测试与离线标定 JSON 已包含。
- 物理 MuJoCo 仿真需在你的 Windows 已安装环境运行；本次环境缺少 `mujoco` 包，未在这里证实物体是否达到新标准。

## 重要：两个评价体系同时存在

| 字段 | 新旧 | 严格含义 |
|---|---|---|
| `contact_success_proxy` | 旧 | 任务接触达到旧12帧门槛，允许夹具辅助接触获取 |
| `stable_contact_proxy` | 旧 | 双指接触 + 相对**初始世界姿态**的偏差 <= 5°（原始每帧列） |
| `legacy_stable_success_proxy` | 旧 | 旧指标曾经连续满足12帧，永久记录为True（便于历史对照） |
| `legacy_stable_frame_ratio` | 旧 | 旧5°指标的活跃帧比例 |
| `hold_stable_proxy` | **新** | 物体已释放夹具 + 连续接触 + 相对手掌完整位姿的滚动1秒窗口变化满足阈值 |
| `stable_success_proxy` | **新** | 最后一个有效任务帧的 `hold_stable_proxy`，在有效任务阶段丢失接触/超阈值会恢复False |
| `ever_hold_stable_proxy` | 新 | 有效阶段曾有过至少一个满足要求的完整1秒窗口；仅用于诊断，不能等同最终成功 |
| `stable_frame_ratio` | **新** | 新指标在有效任务帧中的比例（不可与旧实验数值直接当同一指标比较） |
| `hold_relative_translation_span_m` | 新 | 1秒窗口内物体相对手掌三维位置的最大**成对欧氏距离**（m） |
| `hold_relative_rotation_span_deg` | 新 | 1秒窗口内物体相对手掌姿态的最大**成对旋转角差**（°） |
| `hold_window_samples` | 新 | 截至该帧已连续积累多少个有效物理接触样本，60Hz满1秒需61帧 |
| `hold_object_free` | 新 | 此帧物体不受接近阶段的临时夹具固定、不处于重置帧，且任务活跃 |

***新稳定是“持续接触与相对位姿变化较小”的代理指标，不是接触力验证、无滑移证明、抗扰动或提举成功。***

## 可调阈值（需要根据实验标定，不可任意调松制造成功）

`configs/simulation.yaml`:

```yaml
mujoco:
  stability_orientation_deg: 5.0
  evaluation:
    min_success_streak_frames: 12
    hold_window_s: 1.0
    hold_max_translation_m: 0.001
    hold_max_rotation_deg: 1.0
```

窗口取连续至少61帧（60Hz，覆盖至少1秒）。该窗口的三维平移和旋转**所有时刻两两比较**，有一对超阈值则此帧为False。物体放开夹具后即开始积累：如果初期发生调整，后续滚动窗口仍有机会变得稳定。如果发生接触丢失、物体重置或夹具介入，立即清空窗口；恢复接触后重新积累满一秒。

## 运行方法（PowerShell）

**1. 解压到新目录，进入工程根目录。不要覆盖旧工程。**

建议：`D:\Vhand\CHM6_StableGrasp_Full_Project\Virtual-Dexterous-Hand-main`。

**2. 将旧目录的诊断文件复制到新目录**：

```powershell
$old = 'D:\Vhand\CHM6_TEQS_SchemeE_Full_Project\Virtual-Dexterous-Hand-main'
Copy-Item -LiteralPath "$old\数据手套数据\CHM6_pinch_hold_diagnostic_only.xlsx" -Destination '.\数据手套数据\' -ErrorAction Stop
```

**3. 使用旧工程的 Python 虚拟环境解释器，执行新代码**（无需把 `.venv` 打包到ZIP里）：

```powershell
$python = 'D:\Vhand\CHM6_TEQS_SchemeE_Full_Project\Virtual-Dexterous-Hand-main\.venv\Scripts\python.exe'
& $python -m pytest -q test\tests_20261012\test_relative_grasp_stability.py test\tests_20261012\test_task_evaluator.py test\tests_20261012\test_tqhg_online.py
& $python -m src.main --input glove --glove-samples '数据手套数据\CHM6_pinch_hold_diagnostic_only.xlsx' --sim mujoco --task pinch --steps 360 --run-name stable_v2_tqhg_hold --mapping-method tqhg --tqhg-calibration configs/tqhg_online_v1.json --coupled-servo-ablation hold --servo-trace
```

想看Viewer，在第二条末尾加 `--render --camera closeup --realtime`。

**4. 一键跑五种模型**（先确认前述单组运行正常）：

```powershell
& '.\run_stability_comparison.ps1'
```

结果保存在新项目 `outputs/stable_v2_<method>_hold_pinch.csv` 和 `_summary.json`。新的 `stable_success_proxy` 含义已改变，不要拿它与旧的 `stable_success_proxy` 字段直接比较；请使用新数据的 `legacy_stable_success_proxy` 对照历史。

## 自测边界

- 已验证纯Python稳定窗口单元测试、评价状态机回归测试、TQHG/TEQS相关单元测试及无MuJoCo的启动和CSV写入。
- 未验证完整Windows物理抓握试验；模拟物体质量、摩擦、手指接触和动力学是否足以真实稳定仍需实验。
- 报告接触成功需说明夹具辅助阶段与自由物体阶段；自由物体接触帧计数保存在 `free_object_contact_frames`。
