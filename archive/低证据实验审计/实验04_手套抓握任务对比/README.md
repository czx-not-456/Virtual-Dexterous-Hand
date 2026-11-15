> 已退出核心消融列表：个性化回放结果不一致、完整模型检查点和独立任务验证不足；作为补充审计保留。 本目录是历史可追溯材料，**不代表独立验证结果**。

# 基础映射与个性化映射：抓握任务对比

**实验目标**：PINCH/WRAP 任务中的基础映射与个性化映射配对回放。

**已有结果**：results/ 内有 frames.csv、summary.csv、summary.json（来自 test 历史结果副本）。

## 本文件夹放什么

- 本实验的实际执行脚本（直接位于本目录）；旧命令在 `scripts/`、`experiments/` 或仓库根目录保留兼容入口。
- `results/`：本实验已有结果；新运行结果默认存入 `results/generated/`（默认不纳入 Git）。
- `README.md`：本实验的用途、结果状态与运行方法。

详细评价定义见 [`GLOVE_TASK_COMPARISON.md`](GLOVE_TASK_COMPARISON.md)。

## 运行（从仓库根目录）

```powershell
python experiments/实验04_手套抓握任务对比/compare_glove_tasks.py --help
```

**解释边界**：真实仿真回放需要 MuJoCo；已有数据与重新运行的结果应区分。

共享算法 `src/`、模型配置 `configs/`、原始数据 `数据手套数据/` 与原始测试 `test/` 仍在仓库原位置，不复制共享代码。
