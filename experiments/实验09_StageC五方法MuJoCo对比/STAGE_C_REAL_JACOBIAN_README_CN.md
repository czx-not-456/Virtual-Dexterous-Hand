# CH-M6 阶段 C：真实任务雅可比下的双权重消融及 MuJoCo 显式传感器缺失测试

## 目的与状态

本工程在 **阶段 B 完整工程**基础上增加阶段 C，保留 `tqhg`、`tqhg-v2`、TEQS、原 Hold 接触伺服器、StableGrasp-v2 评价器和全部既有模型，不覆盖任何旧实验基线。

**已在构建环境验证**：历史 124 TRAIN / 66 CAL 数据分离、Stage C 校准生成、Python 单元和历史回归测试、五种算法 `--sim none` 启动/CSV记录、故障时序/恢复与安全回退测试。

**尚未验证**：构建环境没有 `mujoco` 包，不能证明 Stage C 的真实 MuJoCo 抓握改善。需要在使用者的 Windows MuJoCo 中实跑。旧 `CHM6_pinch_hold_diagnostic_only.xlsx` 360帧文件不在压缩包中，需要从旧项目复制。当前实验数据仍不支持 TQHG 超过强单专家的结论。

## 方法（相同数据与模型条件）

| 方法 | 映射 | 任务门控 α | 互补门控 β |
|---|---|---|---|
| `robust6` | 公平增强训练的 30→6 Ridge6 | 不使用 | 不使用 |
| `robust4` | 同训练增强的 30→4→6 协同 | 不使用 | 不使用 |
| `stagec-alpha` | 双专家 | 条件化标定 | 固定0 |
| `stagec-beta` | 双专家 | 固定0 | 条件化标定 + 辅助姿态软约束 |
| `stagec-dual` | 双专家 | 条件化标定 | 条件化标定 + 辅助姿态软约束 |

所有方法：相同6→11关节耦合、相同速度限幅与 `--coupled-servo-ablation hold`。动态任务投影由当前MuJoCo位姿计算：

`J = J_tip(q_actual) @ C`, `P = J⁺J`（数值截断），`N = I-P`。

融合候选：`u = u_A + alpha*P@(u_B-u_A) + beta*N@(u_B-u_A)`。实际执行还受盒约束、限速及接触伺服影响，不能把局部零空间性质直接当作最终物理动作保证。

训练/校准数据：历史124条用于拟合两个专家，另66条仅用于风险标定，分别生成0/1/3/5路缺失的相同掩码误差对。**未读取之前24帧测试文件与360帧诊断捏合文件进行拟合**。门控允许优化到0；α、β不人为设置正下界。

## Windows 工作步骤（推荐解压完整工程到新目录）

1. 将完整 ZIP 解压到独立新目录（不要覆盖当前可用工程）。进入 `Virtual-Dexterous-Hand-main` 根目录。
2. 设置已安装好 MuJoCo 的 Python：

```powershell
$python = "D:\Vhand\CHM6_TEQS_SchemeE_Full_Project\Virtual-Dexterous-Hand-main\.venv\Scripts\python.exe"
```

3. 从**你本机已有**稳定抓握项目复制360帧输入：

```powershell
$old = "D:\Vhand\CHM6_StableGrasp_v2_TQHG_Full_Project\Virtual-Dexterous-Hand-main"
Copy-Item "$old\数据手套数据\CHM6_pinch_hold_diagnostic_only.xlsx" ".\数据手套数据\" -ErrorAction Stop
```

4. 运行相关测试（可用与工程无关的 Python 环境）：

```powershell
& $python -m pytest -q test\tests_20261012\test_stageC_task_gating.py
```

5. 如需自行重建**仅使用历史校准数据**的门控模型：

```powershell
& $python .\scripts\calibrate_stageC_task_gate.py
```

压缩包已包含已校准的 `configs/stageC_task_gate.json`，通常无需重新训练或标定。

### 先单独可视化 TQHG 阶段C双门控（不注入故障）

```powershell
& $python -m src.main `
  --input glove `
  --glove-samples "数据手套数据\CHM6_pinch_hold_diagnostic_only.xlsx" `
  --sim mujoco --render --realtime --camera closeup `
  --task pinch --steps 360 `
  --mapping-method stagec-dual `
  --stagec-robust6-model configs/stageB_robust_ridge6.json `
  --stagec-gate-model configs/stageC_task_gate.json `
  --tqhg-v2-calibration configs/tqhg_online_v2.json `
  --coupled-servo-ablation hold --servo-trace `
  --run-name stageC_visual_clean
```

### 指定第120～239帧第1、4、7路传感器缺失（从第240帧恢复）

在上面命令末尾，增加以下三项（并给一个不同 `run-name`）：

```powershell
  --stagec-fault-channels "1,4,7" `
  --stagec-fault-start 120 `
  --stagec-fault-stop 240
```

含义：故障通道编号从1开始，不是Python的0起始索引；**只把对应原始传感器标记为无效，并传给两专家相同掩码**；并不会改变 `glove` 人手意图识别源数据。**这只是映射层故障注入，不是整套数据手套信号链的端到端失效试验**。

### 自动5种方法×两种场景（建议关闭渲染）

```powershell
& $python .\scripts\run_stageC_mujoco.py `
  --glove "数据手套数据\CHM6_pinch_hold_diagnostic_only.xlsx" `
  --conditions clean,mask_three `
  --task pinch --steps 360
```

自动生成10次仿真，在 `outputs/stageC_mujoco` 输出10份CSV、10份JSON与一份 `stageC_summary.csv`。如果还想测试1路和5路缺失，设置：

```powershell
--conditions clean,mask_one,mask_three,mask_five
```

比较时严格使用每种方法同一组通道组合、同一开始和恢复帧、同一控制器，禁止把某种方法单独改成更有利的故障通道。

## 日志字段

- `mapping_visible_rank`：真实MuJoCo任务投影的数值秩（`--sim none` 则为0）。
- `mapping_w_task / mapping_w_null`：任务方向α、互补方向β。
- `mapping_task_delta_norm / mapping_null_delta_norm`：融合前的任务/互补方向修正范数。
- `mapping_stagec_projection_leakage`：归一化 `||J N||`，检查任务投影零空间数值可靠性。
- `mapping_stagec_task_output_shift`：**融合+边界处理后**的 `||J(u_out-u_A)||`；此数值不等于真实指尖最终误差。
- `stagec_fault_active / stagec_fault_count`：显式故障状态及通道数。
- `mapping_invalid_channels`：实际有效性检测为不可靠的通道数（可能多于显式故障数）。
- `q_*`：真实MuJoCo关节状态；可与鲁棒Ridge6进行逐帧对齐差异分析。
- `hold_stable_proxy`：StableGrasp v2相对手掌的滚动一秒稳定代理，**不是**无滑移提举/受扰动抓握的力学证明。

`stageC_summary.csv` 还包含故障窗口内的接触比例、指尖误差、稳定窗口通过比例，以及恢复后一段的接触比例。由于当前诊断动作的意图持续时间有限，不要把松开动作后的空窗混入故障保持段。

## 严谨性限制与下一阶段

1. 当前配对实验只有一个360帧诊断轨迹，重复运行不等于独立试验；要增加新的连续动作与不同物体初态。
2. 当前阶段C专家训练基于软件教师6维标签，不是真实关节测量；可部署性的验证仍需额外数据。
3. 主动缺失掩码是明确给定的，不能声称已经自动识别真实传感器卡死。
4. `--sim none` 的任务方向P=0，α不代表真实任务门控；必须通过MuJoCo输出的非零任务秩验证C阶段核心机制。
5. 单一捏合Hold较强，接触率接近饱和；未来还要增加独立屈指、圆柱包络、多独立episode和更丰富的外部测试数据。

**研究决策标准**：若双门控与仅α/β相比没有优势，应保留强鲁棒6维专家，不应人为调高门控或改变成功阈值来制造正结果。
