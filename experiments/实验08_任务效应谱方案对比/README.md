> **核心清单精简说明：** 已保留：将动态选维与固定 6 维区分，并提供逐姿态效率及精度权衡。 统一运行与统计说明见 [核心实验导航](../映射对比实验导航.md)。

# 方案 E 任务效应谱映射对比

**实验目标**：方案 E（TEQS）与其对照方案的配对离线对比。

**已有结果**：results/schemeE_paired_comparison.json 为已有结果。

## 本文件夹放什么

- 本实验的实际执行脚本（直接位于本目录）；旧命令在 `scripts/`、`experiments/` 或仓库根目录保留兼容入口。
- `results/`：本实验已有结果；新运行结果默认存入 `results/generated/`（默认不纳入 Git）。
- `README.md`：本实验的用途、结果状态与运行方法。

## 推荐运行：方案 E 深化消融（重点实验）

从仓库根目录运行：

```powershell
python experiments/benchmark_scheme_e_research.py --dry-run
python experiments/benchmark_scheme_e_research.py --steps 24 --repeats 3 --out outputs/scheme_e_research
```

该实验严格拆分三个问题：

1. **动态选维 vs 固定 k=4/5/6，并额外扫严格/宽松谱截断预算**：使用相同 E 控制链路，只改变维数选择规则。
2. **解析有界步进 vs 小型 QP**：使用相同动态选维，比较求解方式对精度和时延的影响。
3. **最终非线性守卫开/关**：评估守卫对最终 FK 误差、回退与输出的影响；关闭守卫只用于诊断，不作为部署建议。

另含 `original_direct` 与 `fixed_PCA4` 两个背景基线。Direct 11-DoF 与 E 六维耦合空间不同，计时仅作描述性比较。

输出目录包含 `report.md`、`aggregate.csv`、`run_summary.csv`、`per_frame_metrics.csv`、`mode_selection.csv`、`paired_pose_deltas.csv`、`paired_comparisons.csv`、`summary.json` 和 `manifest.json`。

**重要指标定义**：
- `requested_k`：选择器请求的谱维数。
- `accepted_k`：实际接受的谱方向数，若回退则为 0。
- `output_rank`：最终输出控制空间秩。
- `fallback_rate`：回退到可行参考命令的比例。

旧字段 `spectrum_modes` 表示输出命令秩，不再用作“动态选维数量”。

## 结果解释边界

离线标签 MAE 不是真实机器人关节真值。重复运行只用于估计计算时延波动，不构成新增独立样本。当前 24 帧/6 姿态的姿态簇 Bootstrap 仅作探索性敏感度分析，不是显著性证明。方案 E 的局部谱分解结论只适用于代码中定义的局部线性模型和固定谱字典，不等于全局最少自由度定理。


## 本轮研究结果与推荐解读

当前已附入一份 24 帧、6 姿态、2 次计算重复的探索性重跑：`results/generated/方案E深化消融_24帧_2次重复/report.md`。请先读 [方案 E 研究深化计划与当前结论](方案E研究深化计划与当前结论.md)。

本轮最重要的发现不是“动态选维已成功”，而是动态选择器存在请求维度与实际输出秩的落差：当前 E_dynamic 请求 k<6 的帧占 91.7%，最终实际输出秩<6 仅 16.7%，回退率 37.5%。固定 k=6 的 MAE 略低于动态 E，且无回退。因此下一阶段优先分析回退原因，再优化谱预算；不要直接放宽守卫来追求较高降维率。

正式复跑命令：

```powershell
python experiments/benchmark_scheme_e_research.py --dry-run
python experiments/benchmark_scheme_e_research.py --steps 24 --repeats 3 --out outputs/scheme_e_research
```

完整运行使用同一 24 帧数据的重复计算主要估计计时波动，不能增加独立样本量。论文结论仍需独立采集批次和未见姿态验证。
