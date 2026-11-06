# 方案 C 与方案 D 可信度映射对比

**实验目标**：方案 C、方案 D 及 D 消融条件下的离线标签对比。

**已有结果**：results/schemeD_comparison.json 为已有结果。

## 本文件夹放什么

- 本实验的实际执行脚本（直接位于本目录）；旧命令在 `scripts/`、`experiments/` 或仓库根目录保留兼容入口。
- `results/`：本实验已有结果；新运行结果默认存入 `results/generated/`（默认不纳入 Git）。
- `README.md`：本实验的用途、结果状态与运行方法。

## 运行（从仓库根目录）

```powershell
python experiments/benchmark_credible_synergy.py --help
```

**解释边界**：该小样本结果不能证明真实机器人上的显著性能提升。

共享算法 `src/`、模型配置 `configs/`、原始数据 `数据手套数据/` 与原始测试 `test/` 仍在仓库原位置，不复制共享代码。
