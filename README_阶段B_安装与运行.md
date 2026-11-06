> **整理说明（2026-11-06）：** 阶段B源文件、结果现统一收于 [`实验01_StageB七方法消融/`](实验01_StageB七方法消融/)；下述 `scripts/` 入口仍兼容执行。历史报告中原有的输出位置示例保留。

# CH-M6 TQHG 阶段B：同等增强训练与七组离线消融

## 目的

检验早期“鲁棒4维专家优于原始Ridge6”是否主要来自**掩码/缺失增强训练**，并分离出模型维数和融合机制的影响。本包不会修改 `src/main.py`、`src/retargeting/`、Hold控制器、MuJoCo评价器，也**不会覆盖**既有 `tqhg_online_v1.json` / `tqhg_online_v2.json`。

此包是**离线**消融；`ridge6_robust` 暂未注册成 `--mapping-method`，不可直接用于MuJoCo，必须通过后续单独在线集成。

## 目录

```text
scripts/stageB_common.py                训练与测试统一定义
scripts/train_stageB_robust_ridge6.py   30维掩码增强6维Ridge训练；只读历史124/66分割
scripts/evaluate_stageB_ablation.py    固定独立24帧，配对相同故障掩码测试七种方法
configs/stageB_robust_ridge6.json      本包预训练可复核模型；可用训练命令重建
 test/tests_20261012/test_stageB_ablation.py   逻辑与端到端测试
实验01_StageB七方法消融/results/published_example/              示例CSV、JSON、报告和图片
```

## Windows 安装（以你之前的工程为例）

当前工程：

```text
D:\Vhand\CHM6_TQHG_v2_Robust_Mapping_Full_Project (1)\Virtual-Dexterous-Hand-main
```

**先复制备份现有工程。** 若使用“增量补丁ZIP”，解压后把其中的 `scripts`、`configs`、`test` 文件夹复制到上面项目根目录，选择**合并文件夹**，不要删除原文件夹。这些文件都是新增文件名，无需替换原算法文件。`examples` 可选拷贝，不影响执行。

若下载的是“完整工程ZIP”，直接解压到**另一个**目录，例如：

```text
D:\Vhand\CHM6_TQHG_StageB_Full_Project\Virtual-Dexterous-Hand-main
```

## 运行（复制以下PowerShell命令）

```powershell
cd "D:\Vhand\CHM6_TQHG_v2_Robust_Mapping_Full_Project (1)\Virtual-Dexterous-Hand-main"
$python = "D:\Vhand\CHM6_TEQS_SchemeE_Full_Project\Virtual-Dexterous-Hand-main\.venv\Scripts\python.exe"

# 1. 测试，不依赖安装MuJoCo包
& $python -m pytest -q test\tests_20261012\test_stageB_ablation.py

# 2. 使用历史124帧训练与66帧校准，单独生成新模型（不覆盖v1/v2）
& $python .\scripts\train_stageB_robust_ridge6.py

# 3. 固定20261025随机种子，沿用24帧测试与20组配对掩码
& $python .\scripts\evaluate_stageB_ablation.py `
  --test ".\数据手套数据\captured_samples(1).xlsx" `
  --model-v2 ".\configs\tqhg_online_v2.json" `
  --robust-ridge6 ".\configs\stageB_robust_ridge6.json" `
  --trials 20 `
  --seed 20261025

# 4. 直接查看中文报告
Get-Content ".\outputs\stageB_ablation\REPORT.md" -Encoding UTF8
```

如果使用全新解压的完整工程，第一条 `cd` 改成其真实新目录即可；原项目 `.venv` 可以继续调用。

## 方法说明

| 方法 | 输入 | 训练 | 说明 |
|---|---|---|---|
| ridge6_original | 插补15维 | 历史无增强 | 原始Ridge6基线 |
| pca4_original | 插补15维 | 历史无增强 | 原PCA4基线 |
| ridge6_robust | 15传感器+15有效掩码 | **124条×6次（744）**、正则化2.0 | 新的公平增强6维模型 |
| pca4_robust | 同上 | **同一124条×6次（744）**、正则化2.0 | 现有TQHG-v2的4维专家 |
| fixed_50_50 | 两个增强专家输出 | 权重固定0.5 | 静态融合基线 |
| calibrated_scalar_blend | 两个增强专家输出 | **仅66条校准数据** | 条件化标量混合，不使用任务雅可比 |
| tqhg_v2_original | 原TQHG-v2 | 原有124/66 | v2离线状态，`P=0,N=I` |

所有模型在同一测试故障掩码下配对评估。错误度量是**归一化6维软件教师标签的MAE**，不是机械手实际运动的真值。

## 输出

`outputs/stageB_ablation/`：

- `paired_cases.csv`：每条测试记录与每组故障掩码下的七种方法误差，以及增强6D/4D专家胜者；
- `summary_by_condition.csv`：按正常/缺失1、3、5路汇总；
- `summary_by_pose.csv`：按六类姿态分组；
- `paired_pose_bootstrap.csv`：仅以**六类姿态**作为聚类单位的探索性自助法区间（不是正式显著性结论）；
- `summary.json`、`REPORT.md`：汇总和解释；
- `mae_by_condition.png`、`pose_error_distribution.png`：结果可视化。

## 本次示例结果（不要事后选取有利数字）

| 缺失情况 | 增强Ridge6 | 增强4维 | 校准混合 |
|---|---:|---:|---:|
| 正常 | **0.043031** | 0.046233 | 0.043031 |
| 缺失1路 | **0.056774** | 0.058701 | 0.056774 |
| 缺失3路 | **0.086659** | 0.087601 | 0.086705 |
| 缺失5路 | **0.121878** | 0.122121 | 0.121880 |

**研究结论**：此批测试的总体MAE最优是公平增强Ridge6；现有鲁棒4维相对原Ridge6的改善不能证明源自4维协同；当前融合也未超过公平增强6维单专家。这是负面但重要的消融结果。

## 有效性边界

1. `captured_samples(1).xlsx` 已用于之前的阶段A探索；**不是一个新的、未被查看过的盲测集**。
2. 测试仅24帧、6种预设姿态。20次掩码重复是同一批姿态的扰动，并非20次独立人体采集。
3. 按6姿态分组的bootstrap区间是探索性描述，样本簇数太少，不用于强显著性宣称。
4. 所谓Oracle（按每个样本测试标签在两个增强专家间选误差较小者）只能诊断潜在互补，不能作为在线方法。
5. 无MuJoCo任务投影时 `P=0,N=I`；本实验**无法证明TQHG双权重α/β的任务分离作用**。此目标属于后续多任务MuJoCo消融。
6. 当前软件教师标签不是独立人手或机器人真值；需要新采连续手套数据和力学抓握评估。

下一步应先获得未用于方案选择的新测试批次，再考虑新任务条件门控，而非单纯提高混合权重。
