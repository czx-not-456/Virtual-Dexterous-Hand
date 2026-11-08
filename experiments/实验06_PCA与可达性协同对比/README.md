# PCA、方案 B 与方案 C 可达性对比

**实验目标**：固定 PCA、方案 B 自适应协同和方案 C 可达性门控等的离线对比。

**已有结果**：results/reachability_5way_offline_24frames.json 为已有结果。

## 本文件夹放什么

- 本实验的实际执行脚本（直接位于本目录）；旧命令在 `scripts/`、`experiments/` 或仓库根目录保留兼容入口。
- `results/`：本实验已有结果；新运行结果默认存入 `results/generated/`（默认不纳入 Git）。
- `README.md`：本实验的用途、结果状态与运行方法。

## 运行（从仓库根目录）

```powershell
python experiments/实验06_PCA与可达性协同对比/benchmark_reachability_synergy.py --help
```

**解释边界**：24 帧预设姿态的软件标签结果，不是抓握成功率。

共享算法 `src/`、模型配置 `configs/`、原始数据 `数据手套数据/` 与原始测试 `test/` 仍在仓库原位置，不复制共享代码。
