> **核心清单精简说明：** 已保留：通过统一运行脚本读取相同姿态标签，计算固定 PCA4 vs B 的成对误差。 统一运行与统计说明见 [核心实验导航](../映射对比实验导航.md)。

# PCA 与方案 B 自适应协同对比

**实验目标**：固定 4 维 PCA 与方案 B 自适应协同的离线标签诊断。

**已有结果**：`results/SCHEME_B_VALIDATION.json` 保存了旧版非 MuJoCo 24 帧诊断，不是完整的独立新运行 benchmark，更不能证明抓握性能。

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
