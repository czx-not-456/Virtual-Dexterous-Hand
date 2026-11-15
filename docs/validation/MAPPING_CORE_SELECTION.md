# 映射消融精简与证据等级（候选整理）

## 目标

仅讨论映射算法的误差、故障鲁棒性、约束/风险控制与优化器开销；不将 WRAP 接触检测/保持控制与映射创新混为一谈。历史结果的好坏**不是删掉实验数据的依据**，缺少可审计数据或比较对象不成立才移出核心列表。

## 主线决定（保留原编号便于溯源）

| 原编号 | 处理 | 理由与下一步 |
|---|---|---|
| 01 Stage B | 核心 A | 同一缺失掩码的配对消融已能复跑；补充独立新数据与一致训练条件基线 |
| 05 B 自适应 | 核心 B | 小幅改善；统一姿态配对比较，并报告优化开销 |
| 06 C 可达性 | 核心 C | 离线 MAE 可能下降或上升，需额外报告实际可行性/约束指标，不能用 MAE 判死刑 |
| 07 D 可信度 | 核心 D | 不确定性模块消融存在零贡献；保留其负结果，继续隔离非线性守卫贡献 |
| 08 E 任务效应谱 | 核心 E | 加速值得验证，动态选维与固定全 6 维差异极小；分开验证选维与求解器 |
| 09 Stage C | 待执行验证 | 有真实 MuJoCo 执行入口，但本次没有形成新的成对物理数据，不当作成功案例 |

## 移出的实验（仍在存档）

- 原 02：Mock 静态/动态演示；**不是独立真实映射对照**。
- 原 03：个性化标定拟合结果；只有很少验证姿态，作为补充，不作泛化证明。
- 原 04：个性化抓握回放，PINCH/WRAP 不一致，模型完整性/可复现性尚不足；作为补充。
- 原 10：缺少完整成对运行结果，与 Stage C 验证目标重叠；保留历史脚本和说明在存档。

目录：[`archive/低证据实验审计/`](../../archive/低证据实验审计/README.md)。代码、数据没有被物理销毁，旧兼容入口仍可读取存档脚本。`test` 中仍保留相关审计测试。

## 本次实际代码优化

1. 新增 `experiments/run_core_mapping_suite.py`：统一入口、相同样本路径/参数、`--dry-run`、独立原始运行日志、输入 SHA256 与机器环境 manifest；失败时保留日志，不写虚假的成功报告。
2. 新增 `experiments/core_mapping_evidence.py`：对每个生成的逐帧 CSV 检查帧数与索引，逐帧计算教师标签误差，按预设姿态聚类配对 bootstrap；输出 `evidence_report.md`、`paired_comparisons.csv`、`evidence.json`。
3. 方案 E 新增 `--samples`、`--steps`、`--out`、`--modes` 参数，可只保留动态 vs 固定全 6 维及直接基线，避免无意义重复调用 C/D；保留旧版完整运行入口兼容性。
4. 修复方案 E 在主程序中的接口缺失：之前 `set_object_task_goal` 未定义、多个诊断字段未定义；修复后不再把未完成坐标系对齐的 object-space 诊断伪造成有效数字。
5. 原实验移出主目录，保留历史结果/模型/共享实现；没有重写或删除不利结果。

## 证据边界与公平性

- **历史测试文件 `captured_samples(1).xlsx` 只有 24 帧、6 姿态**，且已反复查看。不是独立盲测；每组姿态 4 帧也不是 24 名独立受试者。新增配对 bootstrap 只用于观察姿态敏感度，不可宣称总体显著性。
- Stage B 所用归一化六维 MAE 与 B/C/D/E 的角度 MAE 不是同一量纲；不可直接排名。
- Stage B **增强模型和旧模型的训练处理不同**，改善不能单独归因于 PCA/Ridge 维度。
- B/C/D 不同模块的指标与约束可能不同；应同时考察可行率、风险/守卫激活、误差和时延。
- E 动态与 E 固定 6 维是有效的选维消融；E 与原直接 11-DoF 优化的空间不同，约束与计算任务不完全同等，速度差是工程差异，不是理论必然优势。
- 由于旧版程序 E 的运行接口曾缺失，本次修复后重跑离线数据；**不能把旧存档的 E JSON 直接当成当前代码的执行证明**。
- Stage C 需 Windows 的 MuJoCo 环境继续运行；本次交付不含虚构的 Stage C 结果。物理接触/抓握指标不在当前核心离线证据范围内。

## 复现入口（Windows PowerShell）

```powershell
# 在项目根目录；已有 .venv 的用户可直接激活自己的环境
python -m pytest -q test/tests_20261012/test_core_mapping_suite.py test/tests_20261012/test_task_effect_spectrum.py
python experiments/run_core_mapping_suite.py --dry-run
python experiments/run_core_mapping_suite.py --groups stageb,modules,spectrum --steps 24 --out outputs/mapping_core
Get-Content outputs\mapping_core\evidence_report.md -Encoding UTF8
# 若耗时任务已经写完全部 CSV 但统一流程中途退出，可验证现有 CSV 并重新生成报告：
python experiments/run_core_mapping_suite.py --groups stageb,modules,spectrum --steps 24 --out outputs/mapping_core --report-only
# Stage C 先检查真实物理仿真命令
python scripts/run_stageC_mujoco.py --dry-run
```

`--report-only` 会核对 manifest 的输入哈希、配置的样本数/分组和新报告所需 CSV 的每帧索引，**它不能证明中断的子进程 exit code 为 0**。

## 研究建议

在新独立采集序列上重新做配对、控制超参数与试验预算，优先回答：

1. 鲁棒训练是否在同等增强/训练量下仍提升故障泛化？
2. C 的 reachability gate 是否带来约束可行性、任务误差或恢复性能，而非只改变教师标签误差？
3. D 的不确定性组件是否必须？如消融继续为零，应简化或删除该组件，而不是隐藏负结果。
4. E 的动态维度选择在更广泛姿态/扰动分布下，是否能真正经常选择 k<6，是否有额外加速？
