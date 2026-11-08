# PCA 与方案 B 自适应协同对比

**实验目标**：固定 4 维 PCA 与方案 B 自适应协同的离线标签诊断。

**已有结果**：当前包未找到该脚本单独保存的历史结果。

## 本文件夹放什么

- 本实验的实际执行脚本（直接位于本目录）；旧命令在 `scripts/`、`experiments/` 或仓库根目录保留兼容入口。
- `results/`：本实验已有结果；新运行结果默认存入 `results/generated/`（默认不纳入 Git）。
- `README.md`：本实验的用途、结果状态与运行方法。

## 运行（从仓库根目录）

```powershell
python experiments/实验05_PCA与自适应协同对比/benchmark_adaptive_synergy.py --help
```

**解释边界**：仅离线软件教师标签指标，不等同于实际接触/抓握表现。

共享算法 `src/`、模型配置 `configs/`、原始数据 `数据手套数据/` 与原始测试 `test/` 仍在仓库原位置，不复制共享代码。
