# CH-M6 TQHG-v2：鲁棒4维专家 + 任务/互补双权重映射

## 1. 包含哪些内容

本工程以**StableGrasp v2 + Hold + 在线 TQHG-v1**归档为基础制作，添加了可单独选择的 `tqhg-v2`，并**保留**原有 `ridge6`、`pca4`、`fixed-blend`、`single-gate`、`tqhg`、`legacy`、TEQS 和新旧稳定评价字段。

- `src/retargeting/tqhg_v2.py`：掩码增强鲁棒4维专家、保守传感器可信度评估、条件化任务权重α/互补权重β、上一帧平滑、重度缺失回退。
- `src/retargeting/tqhg_online.py`：只增加嵌入式v1模型兼容，原v1路径/映射公式保留。
- `src/main.py`：增加 `--mapping-method tqhg-v2`、`--tqhg-v2-calibration`，日志增加8个 v2 诊断列。
- `scripts/train_tqhg_v2.py`：历史数据前124帧训练（六种姿态内顺序分割），剩余66帧用于门控标定；独立随机种子生成合成缺失案例。**不会读入24帧外部测试或360帧捏合诊断数据**。
- `configs/tqhg_online_v2.json`：预先训练并附带的v2模型，可按需重新生成。
- `scripts/evaluate_tqhg_v2_faults.py`：针对独立24帧姿态数据的合成缺失测试。报告的是**软件教师6维标签**的代理误差，并非机器人实测误差。
- `test/tests_20261012/test_tqhg_v2.py`：单元测试。

## 2. Windows 首次运行

**一定解压到新文件夹，不要覆盖上一版本。**

建议：

```text
D:\Vhand\CHM6_TQHG_v2_Robust_Full_Project\Virtual-Dexterous-Hand-main
```

PowerShell：

```powershell
cd "D:\Vhand\CHM6_TQHG_v2_Robust_Full_Project\Virtual-Dexterous-Hand-main"

# 复用你已安装好 MuJoCo 的旧项目 Python
$python = "D:\Vhand\CHM6_TEQS_SchemeE_Full_Project\Virtual-Dexterous-Hand-main\.venv\Scripts\python.exe"
Test-Path $python

# 之前单独构造的360帧捏合诊断输入**不包含在本压缩包内**，从你已有工程复制
$old = "D:\Vhand\CHM6_StableGrasp_v2_TQHG_Full_Project\Virtual-Dexterous-Hand-main"
Copy-Item -LiteralPath "$old\数据手套数据\CHM6_pinch_hold_diagnostic_only.xlsx" -Destination ".\数据手套数据\" -ErrorAction Stop

# 单元测试
& $python -m pytest -q test\tests_20261012\test_tqhg_v2.py test\tests_20261012\test_tqhg_online.py test\tests_20261012\test_relative_grasp_stability.py
```

如果已随包附带 `configs\tqhg_online_v2.json`，**首次运行不用重新训练**。

### 真实MuJoCo可视化（待你本机验证）

```powershell
& $python -m src.main `
  --input glove `
  --glove-samples "数据手套数据\CHM6_pinch_hold_diagnostic_only.xlsx" `
  --sim mujoco --render --realtime --camera closeup `
  --task pinch --steps 360 --run-name tqhg_v2_visual `
  --mapping-method tqhg-v2 `
  --tqhg-v2-calibration configs/tqhg_online_v2.json `
  --coupled-servo-ablation hold --servo-trace
```

**不要**同时添加 `--latent-model` 或 `--task-effect-spectrum`；这样会混淆在线映射比较。

### 5+1个方法公平对照（无渲染）

```powershell
$methods = @("ridge6","pca4","fixed-blend","single-gate","tqhg","tqhg-v2")
foreach ($method in $methods) {
    & $python -m src.main `
      --input glove `
      --glove-samples "数据手套数据\CHM6_pinch_hold_diagnostic_only.xlsx" `
      --sim mujoco --task pinch --steps 360 `
      --run-name "compare_v2_$method" --mapping-method $method `
      --tqhg-calibration configs/tqhg_online_v1.json `
      --tqhg-v2-calibration configs/tqhg_online_v2.json `
      --coupled-servo-ablation hold --servo-trace
    if ($LASTEXITCODE -ne 0) { throw "$method failed" }
}
```

注意：以上6组使用相同Hold，但**单次同一轨迹不能证明统计显著性**。必须补独立轨迹、重复重置与其他任务。

## 3. 如何重新训练v2

```powershell
& $python scripts\train_tqhg_v2.py
```

打印的 `normal/fault CAL proxy MAE` 来自**门控标定集**，不能写成最终论文测试集表现。模型路径为 `configs\tqhg_online_v2.json`；v1的 `configs\tqhg_online_v1.json` 不会被改写。

## 4. 小规模独立故障测试（不使用MuJoCo）

```powershell
& $python scripts\evaluate_tqhg_v2_faults.py --trials 20
```

脚本使用随包原始 `captured_samples(1).xlsx` 的24帧与6维教师标签。故障由随机掩码显式标识。此脚本**没有MuJoCo的实时指尖雅可比**，因此不验证真实任务方向α的操作收益，只用来筛查鲁棒4维分支和输入可靠性逻辑。

## 5. CSV重点诊断字段

基础旧版字段继续保留：`mapping_w_task`、`mapping_w_null`、`mapping_visible_rank`、`mapping_box_clipped`、`hold_stable_proxy`、`stable_success_proxy` 等。

新v2字段：

| CSV列名 | 含义 |
|---|---|
| `mapping_sensor_reliability` | 输入质量可信度估计：0–1 |
| `mapping_invalid_channels` | 明确无效/明显越界的通道数量 |
| `mapping_gate_severity` | 用于正常和合成故障标定风险插值的程度 |
| `mapping_expert_delta_norm` | 鲁棒4维专家和Ridge6的6维输出距离 |
| `mapping_task_delta_norm` | 融合在当前任务可观测方向的改变量 |
| `mapping_null_delta_norm` | 融合在互补方向的改变量 |
| `mapping_post_clip_task_delta_norm` | 裁剪后引入的任务投影差异 |
| `mapping_gate_fallback` | 10路或更多输入无效时触发的保守回退 |

如果正常数据下 α=0，**并不表示Bug**。它可能是经过独立标定后正确选择了Ridge6。关键是验证特定退化情境下，另一分支确实更准确，门控是否做出合理选择。

## 6. 当前局限与必须注意的安全边界

1. **没有自动准确识别传感器卡死**：静止保持动作也会产生恒定读数。对确知缺失的通道，应通过 `sensor_valid_mask` 显式传入；当前 `src.main` 的Excel回放并无来自手套驱动的硬件有效性标记。真实卡死检测必须另行建立并验证。
2. 可信度只使用缺失、越界及快速跳变等保守代理；并非可靠的故障概率估计。
3. 训练、校准只来自单设备六姿态的小数据。尽管独立24帧测试不参与模型训练，但不足以支持跨用户和复杂动态操作泛化。
4. 4维专家受低维瓶颈限制，即使数据退化也不保证优于Ridge6。不能强制设置α/β大于0来制造“算法改进”。
5. `task_projector` 的零空间正交只在当前雅可比线性化下成立。Hold接触反馈、限速、限幅等会使最终真实任务变化偏离局部推断。
6. 新版抓握稳定性仍是**固定基座、释放后物体相对手掌的1秒滚动位姿代理指标**，不等同于夹持力、提举或抗扰动验证。
7. 本打包环境可运行Python算法与无MuJoCo演练，但没有安装MuJoCo；**真实物理抓握和渲染需在你的Windows环境验证**。

## 7. 优先级明确的下一步

(1) 先跑 `tqhg-v2` 的可视化360帧，检查是否动作正确；(2) 跑独立故障测试，比较专家与门控；(3) 统一Hold做6组方法对照；(4) 增加独立屈指和圆柱包络任务，并统计非目标手指串动。**在证据出来前不要宣称v2已优于其他方法。**
