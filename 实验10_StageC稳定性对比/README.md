# Stage C：多方法抓握稳定性对比

**实验目标**：Ridge6、PCA4、融合、单门控、TQHG 等在 PINCH Hold 场景下的稳定性对比。

**已有结果**：当前包未包含完整稳定性对比 CSV；运行需要诊断 Excel 与 MuJoCo。

## 本文件夹放什么

- 本实验的实际执行脚本（直接位于本目录）；旧命令在 `scripts/`、`experiments/` 或仓库根目录保留兼容入口。
- `results/`：本实验已有结果；新运行结果默认存入 `results/generated/`（默认不纳入 Git）。
- `README.md`：本实验的用途、结果状态与运行方法。

在 Windows PowerShell 中运行。脚本会从本目录定位项目根目录，生成结果保存于本实验 `results/generated/stability/`。

## 运行（从仓库根目录）

```powershell
powershell -ExecutionPolicy Bypass -File .\run_stability_comparison.ps1
```

**解释边界**：脚本对诊断输入有依赖，未在本次整理中重跑。

共享算法 `src/`、模型配置 `configs/`、原始数据 `数据手套数据/` 与原始测试 `test/` 仍在仓库原位置，不复制共享代码。
