# 基础映射与个性化映射：标定误差对比

**实验目标**：对比基础映射和个性化映射的角度标定误差。

**已有结果**：results/ 内有 mapping_comparison.json 和逐姿态 CSV（来自 test 历史结果副本）。

## 本文件夹放什么

- 本实验的实际执行脚本（直接位于本目录）；旧命令在 `scripts/`、`experiments/` 或仓库根目录保留兼容入口。
- `results/`：本实验已有结果；新运行结果默认存入 `results/generated/`（默认不纳入 Git）。
- `README.md`：本实验的用途、结果状态与运行方法。

## 运行（从仓库根目录）

```powershell
python experiments/实验03_手套标定映射对比/compare_glove_calibration.py --help
```

**解释边界**：结果副本不替代 test/ 下的原始验收记录。

共享算法 `src/`、模型配置 `configs/`、原始数据 `数据手套数据/` 与原始测试 `test/` 仍在仓库原位置，不复制共享代码。
