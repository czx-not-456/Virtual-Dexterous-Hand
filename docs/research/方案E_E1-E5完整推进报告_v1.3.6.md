# 方案 E（任务效应谱）E1–E5 完整推进报告 — 候选 v1.3.6

## 总览与结论

本轮将 E1–E5 落到代码、实验入口、日志字段和可复现实验数据中。当前最重要的结论是：**动态 E 已在 27 份跨采集记录回放中实现 95.66% 的实际低秩输出，且相对固定 k=6 的教师标签 MAE 几乎不变；但 Pipeline P95 平均慢约 0.52 ms，尚未证明其计算更快。**

因此，方案 E 当前最可信的贡献表述应是“在相近离线标签误差下实现明显降低的输出维度，并满足本次软件管线的 60 Hz 预算”，而不是“精度更高”或“计算更快”。真实连续自由手势、独立受试者泛化和物理 MuJoCo 任务表现仍需要补充。

## E1 — 低维候选拒绝原因诊断

### 代码改动

- 为每个候选维数记录候选尝试次数、谱预算拒绝、编码器保真/速度盒拒绝、非线性 FK/regret 拒绝。
- 记录候选最小 FK 误差、最小 regret、最大编码器保真超额及最终 `selection_status`。
- 区分 `requested_k`（请求维数）、`accepted_k`（通过全部证书的维数）与 `output_rank`（最终实际输出秩），避免将旧字段 `spectrum_modes` 误当成请求维数。
- 修正接受逻辑：候选必须同时通过谱遗漏预算、编码器保真、非线性 FK/regret、速度盒/关节流形可行性；若没有候选同时通过，则回退到可行的完整参考命令。不能仅因某候选的 FK 误差比基线略好，就把它记为谱预算合格的低维候选。

### 24 帧诊断结果

在 24 帧、6 个姿态簇的动态 E 诊断运行中：

- 候选尝试 76 次；谱预算拒绝 0 次；编码器保真/速度盒拒绝 20 次；非线性 FK/regret 拒绝 16 次。拒绝类别可能重叠，不应相加视为互斥样本。
- `requested_k < 6` 占 91.7%；`output_rank < 6` 占 54.2%；fallback 为 0%。
- 说明选择器大多提出低维请求，但受约束检查影响，只有约一半帧最终实际输出低秩命令。

### 跨采集记录诊断

27 份记录中累计统计：编码器保真拒绝 1,598 次、非线性 FK/regret 拒绝 962 次、谱预算拒绝 0 次；动态 E 的实际低秩输出率 95.66%，fallback 率 0%。这是跨记录中观察到的候选诊断统计，不代表独立受试者数量。

## E2 — 任务误差预算 + 可达性 + 非线性约束选维

动态选维的候选接受从单一指标判断改为多条件联合证书：

1. 谱遗漏在配置的任务效应预算内；
2. 编码器保真度满足阈值；
3. 非线性 FK 误差与参考命令 regret 不超过允许范围；
4. 输出速度盒、关节范围和可行性检查通过。

候选维数按由低到高检查，优先接受满足全部条件的最小维数；否则使用可行完整参考命令。各拒绝原因分别记录，允许同一候选出现多个拒绝原因。

本轮没有把风险协方差描述性 proxy 宣称为概率安全保证；当前阈值仍是工程证书，需要在更多任务和独立数据上校准。

## E3 — 动态选维与固定维度公平对照

运行：24 帧、6 个预设姿态，4 种方法，3 次计算重复。所有 E 变体使用相同 E mapper、输入序列、非线性守卫、速度/关节约束和输出评分；只改变维数规则。重复只用于估计运行时间波动，不增加独立样本量。

| 方法 | 教师标签 MAE (°) | 优化器中位数 (ms) | Pipeline P95 (ms) | 实际低秩输出比例 | fallback 比例 |
|---|---:|---:|---:|---:|---:|
| E_dynamic | 16.594 | 5.78 | 9.16 | 54.2% | 0.0% |
| E_fixed4 | 16.633 | 5.52 | 7.41 | 29.2% | 70.8% |
| E_fixed5 | 16.584 | 5.75 | 10.50 | 54.2% | 45.8% |
| E_fixed6 | 16.480 | 5.45 | 7.19 | 0.0% | 0.0% |

解释：

- 本次 24 帧中 E_fixed6 的 MAE 最低；E_dynamic 比它高 0.114°，不能宣称动态 E 精度领先。
- 固定 k=4/5 因可行性与保真约束经常回退，最终实际输出秩并不等于配置的固定 k。不能把它们的标称维数当成实际输出维数。
- E_dynamic 的输出秩均值约 4.708，低秩输出比例 54.2%，且本轮无 fallback。
- 24 帧样本很小，姿态簇 bootstrap 仅作探索性敏感度分析，不是显著性证明。

证据目录：`outputs/e_fair_dimension_controls/`。

## E4 — 跨采集记录留出回放

本轮对项目内全部有效 `captured_samples.csv` 运行了 E_dynamic 与 E_fixed6 配对回放：

- 27 份完整记录，覆盖 12 个手套序列号/左右手组合；两种方法在全部记录上完成，无失败记录。
- 每份记录均作为留出回放文件，不使用该记录拟合映射参数。
- target_1..6 是预设姿态表目标，不是独立测量的机器人关节真值；项目没有独立受试者 ID，因此本实验只能称为跨采集记录/跨手套记录回放，不能称为独立受试者泛化。
- 多次记录可能来自相同序列号；报告同时给出按手套序列号+左右手聚合的配对差值，避免把同一设备重复采集当作独立个体。

### 27 份记录汇总

| 指标 | E_dynamic | E_fixed6 |
|---|---:|---:|
| 记录级 MAE 均值 ± 跨记录 SD (°) | 8.005 ± 2.330 | 7.994 ± 2.323 |
| 优化器中位数的记录均值 (ms) | 5.590 | 5.464 |
| Pipeline P95 的记录均值 (ms) | 8.146 | 7.628 |
| Pipeline P99 的记录均值 (ms) | 9.611 | 9.065 |
| 记录级 16.667 ms deadline miss 比率均值 | 0.000493 | 0.000396 |
| 实际低秩输出比例 | 95.66% | 0% |
| fallback 比例 | 0% | 0% |

配对记录 Dynamic − Fixed6 的 MAE 差均值为 +0.0108°（中位数 +0.0059°；27 份记录中 Dynamic 更低 10 份、Fixed6 更低 17 份）。按 12 个手套序列号/左右手组合等权聚合后，配对 MAE 差均值约 −0.0026°，6 组 Dynamic 更低、6 组 Fixed6 更低。两种分析均指向误差差异非常小，不能宣称精度优势。E_dynamic 的 Pipeline P95 记录均值比 Fixed6 高约 0.52 ms；因此本结果支持降低输出秩且保持相近离线精度，不支持“动态 E 运行更快”。

动态 E 的输出秩分布累计为：k=1：3,870 帧；k=2：192 帧；k=3：202 帧；k=4：332 帧；k=5：521 帧；k=6：232 帧；加权平均输出秩约 1.904。此处低秩输出率高是本轮值得关注的结果，但仍是基于预设姿态标签的记录回放。

实时解释：两种方法的 Pipeline P95 均值均低于 16.667 ms 的 60 Hz 帧预算；动态 E 的 P95 平均慢约 0.52 ms。deadline miss 统计是该软件运行环境的测量值，不构成硬实时保证。

证据目录：`outputs/scheme_e_cross_recording/`，包括 27 份逐记录运行 CSV、汇总表、维数分布、fallback/拒绝原因及配对结果。

## E5 — MuJoCo 任务与实时验证入口

新增/完善 `experiments/benchmark_scheme_e_mujoco.py`，受控比较 E_dynamic 与 E_fixed6，可运行 PINCH 和 WRAP。每种任务/方法输出：

- 任务接触与稳定成功代理；
- 任务空间线性误差均值/P95；
- 伺服误差均值/P95；
- 物体位移；
- Pipeline P95/P99 与 16.667 ms deadline miss；
- 实际低秩输出比例和 fallback 比例。

已通过 dry-run 命令检查，命令规划能生成两任务×两方法的四组仿真调用。**本轮构建环境没有安装 MuJoCo，因此没有在本环境实际执行物理仿真**；请在已有 MuJoCo 4.0.0 的 Windows `.venv` 中运行 E5 命令，不能把 dry-run 说成仿真结果。

Windows 运行命令（项目根目录、已激活 `.venv`）：

```powershell
python experiments/benchmark_scheme_e_mujoco.py `
  --tasks pinch,wrap `
  --modes E_dynamic,E_fixed6 `
  --out outputs/scheme_e_mujoco
```

若需要实时渲染，可增加 `--render --realtime`。首次正式评价建议先不渲染，以减少 GUI 对计时的干扰；随后单独运行可视化。

## 统一复现命令

```powershell
# E1–E3：严格诊断 + 固定维度公平消融
python experiments/实验08_任务效应谱方案对比/benchmark_scheme_e_research.py `
  --steps 24 --repeats 3 `
  --modes E_dynamic,E_fixed4,E_fixed5,E_fixed6 `
  --out outputs/e_fair_dimension_controls_user

# E4：全部 27 份采集记录；支持断点续跑
python experiments/benchmark_scheme_e_sessions.py `
  --modes E_dynamic,E_fixed6 `
  --out outputs/scheme_e_cross_recording `
  --resume

# E5：MuJoCo PINCH/WRAP 对照
python experiments/benchmark_scheme_e_mujoco.py `
  --tasks pinch,wrap --modes E_dynamic,E_fixed6 `
  --out outputs/scheme_e_mujoco
```

## 验证情况与边界

- E 相关单元测试：`test_task_effect_spectrum.py`、`test_core_mapping_suite.py`、`test_wrap_servo_policy_ablation.py`，本轮共 25 项通过。
- E1–E4 的离线运行、E4 27 份记录完整回放及 E5 dry-run 均已完成。
- E5 的实际 MuJoCo 物理运行尚未由本轮构建环境执行；需要在用户 Windows/MuJoCo 环境完成。
- 当前没有独立受试者 ID、自由连续手势真值或硬件实测数据。所有“准确度”指标是相对预设软件教师标签；所有“实时”指标是软件管线测量；不得把它们写成硬件真值、独立受试者泛化或硬实时认证。
