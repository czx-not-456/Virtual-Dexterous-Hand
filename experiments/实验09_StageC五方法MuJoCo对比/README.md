> **核心清单精简说明：** 仅保留为**待完成验证框架**，缺少完整运行结果不能作为已证明贡献。 统一运行与统计说明见 [核心实验导航](../映射对比实验导航.md)。

# Stage C：MuJoCo 五方法对比

**实验目标**：robust6、robust4、仅 α、仅 β、双门控五方法的配对 MuJoCo 对比。

**已有结果**：当前包没有完整的 stageC_summary.csv 和对应成对运行输出。

## 本文件夹放什么

- 本实验的实际执行脚本（直接位于本目录）；旧命令在 `scripts/`、`experiments/` 或仓库根目录保留兼容入口。
- `results/`：本实验已有结果；新运行结果默认存入 `results/generated/`（默认不纳入 Git）。
- `README.md`：本实验的用途、结果状态与运行方法。

附属脚本：`calibrate_stageC_task_gate.py`；`审计工具/` 内为精度、故障、控制路径审计。方法说明见 [`STAGE_C_REAL_JACOBIAN_README_CN.md`](STAGE_C_REAL_JACOBIAN_README_CN.md)。

## 运行（从仓库根目录）

```powershell
python scripts/run_stageC_mujoco.py --dry-run
```

**解释边界**：运行需要 MuJoCo；审计工具/ 是本实验的结果审查工具，不是另一组实验。

共享算法 `src/`、模型配置 `configs/`、原始数据 `数据手套数据/` 与原始测试 `test/` 仍在仓库原位置，不复制共享代码。

## 新增：接触检测判定器消融（候选修订）

四检测器已接入 `--contact-detector`，脚本与说明见 [`接触判定消融/README.md`](接触判定消融/README.md)。本包只包含合成信号示例结果，尚无新产生的 MuJoCo 对比数值，不能声称真实接触成功率提高。
