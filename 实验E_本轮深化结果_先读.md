# 方案 E 本轮深化结果（先读）

本项目的研究主线与实验设计见：`experiments/实验08_任务效应谱方案对比/方案E研究深化计划与当前结论.md`。

本轮实际执行的证据位于：`experiments/实验08_任务效应谱方案对比/results/generated/方案E深化消融_24帧_2次重复/`。

## 最重要的结果

- E_dynamic 的 MAE 为 16.521°，固定 k=6 E 为 16.480°；当前数据没有证明动态选维优于固定 k=6。
- E_dynamic 优化器中位耗时为 6.22 ms，Direct 11DoF 为 24.62 ms；由于控制空间不同，这只是效率—误差背景权衡。
- 48 条动态 E 逐帧记录中，44 条请求 k<6，但最终只有 8 条输出秩<6；18 条发生 fallback（37.5%）。
- 最终非线性守卫触发率为 0%，因此当前数据不能评价守卫被触发时的保护效果。
- 本数据只有 24 帧、6 个预设姿态；重复执行只用于估计计时波动，不能视为新增独立样本。

## 运行

```powershell
python experiments/benchmark_scheme_e_research.py --dry-run
python experiments/benchmark_scheme_e_research.py --steps 24 --repeats 3 --out outputs/scheme_e_research
```

建议下一步先分析 `mode_selection.csv` 中的 fallback 原因和被拒绝的低维候选，再考虑重构谱预算。不要单纯放宽约束以降低 fallback。
