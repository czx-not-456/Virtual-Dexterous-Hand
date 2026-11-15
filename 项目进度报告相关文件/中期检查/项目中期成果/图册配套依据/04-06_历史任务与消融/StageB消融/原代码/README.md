# Stage B：七种映射方法离线消融

**实验目标**：7 种映射算法在正常、缺失 1/3/5 路传感器条件下的配对离线误差对比。

**已有结果**：已有报告、CSV、JSON 和图表（results/published_example/）。

## 本文件夹放什么

- 本实验的实际执行脚本（直接位于本目录）；旧命令在 `scripts/`、`experiments/` 或仓库根目录保留兼容入口。
- `results/`：本实验已有结果；新运行结果默认存入 `results/generated/`（默认不纳入 Git）。
- `README.md`：本实验的用途、结果状态与运行方法。

核心脚本：`stageB_common.py`、`train_stageB_robust_ridge6.py`、`evaluate_stageB_ablation.py`。

## 运行（从仓库根目录）

```powershell
python scripts/evaluate_stageB_ablation.py --help
```

**解释边界**：结果是归一化软件教师标签 MAE，不能当作真实抓握成功率。

共享算法 `src/`、模型配置 `configs/`、原始数据 `数据手套数据/` 与原始测试 `test/` 仍在仓库原位置，不复制共享代码。
