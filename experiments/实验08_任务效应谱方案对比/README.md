# 方案 E 任务效应谱映射对比

**实验目标**：方案 E（TEQS）与其对照方案的配对离线对比。

**已有结果**：results/schemeE_paired_comparison.json 为已有结果。

## 本文件夹放什么

- 本实验的实际执行脚本（直接位于本目录）；旧命令在 `scripts/`、`experiments/` 或仓库根目录保留兼容入口。
- `results/`：本实验已有结果；新运行结果默认存入 `results/generated/`（默认不纳入 Git）。
- `README.md`：本实验的用途、结果状态与运行方法。

## 运行（从仓库根目录，实际执行完整离线对比；不是 `--help` 命令）

```powershell
python experiments/实验08_任务效应谱方案对比/benchmark_task_effect_spectrum.py
```

**解释边界**：离线标签和计时结果不等于真实抓握表现。

共享算法 `src/`、模型配置 `configs/`、原始数据 `数据手套数据/` 与原始测试 `test/` 仍在仓库原位置，不复制共享代码。
