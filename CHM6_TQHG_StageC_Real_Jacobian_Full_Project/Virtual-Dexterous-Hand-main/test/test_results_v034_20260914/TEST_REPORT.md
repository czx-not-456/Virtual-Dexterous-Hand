# Virtual-Dexterous-Hand v0.3.4 自动测试报告

- 测试日期：2026-09-14（Asia/Shanghai，UTC+08:00）
- 项目目录：`F:\大三上\大创\项目文件夹\Virtual-Dexterous-Hand-v0.3.4`
- 结果目录：`test_results_v034_20260914`
- 执行原则：按用户指定顺序执行；单项失败不阻断后续测试；未修改现有源代码或已有 `outputs`。

## 测试环境

| 项目 | 值 |
|---|---|
| 操作系统 | Microsoft Windows NT 10.0.22631.0，64-bit |
| PowerShell | 5.1.22621.3880 |
| Python | 3.11.9 |
| pytest | 9.1.1 |
| MuJoCo | 3.13.0 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |

完整依赖版本见 `environment.txt`。

## 命令与状态

| # | 命令 | 退出码 | 命令状态 | 功能判定 | 日志 |
|---:|---|---:|---|---|---|
| 1 | `pytest -rA` | 0 | 通过 | 28/28 测试通过 | `01_pytest_rA.log` |
| 2 | `python -m src.main --sim none --steps 900` | 0 | 通过 | 900 步运行完成并生成 CSV | `02_sim_none_900.log` |
| 3 | `python experiments/compare_mapping.py` | 0 | 通过 | 脚本完成；WRAP vector MSE 有退化，详见下文 | `03_compare_mapping.log` |
| 4 | `python -m src.main --sim mujoco --steps 1200 --task pinch` | 0 | 通过 | **失败：PINCH 接触目标未达到** | `04_mujoco_pinch_1200.log` |
| 5 | `python -m src.main --sim mujoco --steps 1200 --task wrap` | 0 | 通过 | WRAP 接触代理目标达到 | `05_mujoco_wrap_1200.log` |
| 6 | `python experiments/analyze_contacts.py <PINCH CSV>` | 0 | 通过 | 分析完成 | `06_analyze_pinch.log` |
| 7 | `python experiments/analyze_contacts.py <WRAP CSV>` | 0 | 通过 | 分析完成 | `07_analyze_wrap.log` |

说明：“命令状态”仅表示进程是否正常退出；“功能判定”依据 CSV 指标和项目现有接触验证目标。所有命令均正常退出，但这不等于所有功能目标均通过。

## pytest 结果

- 通过：28
- 失败：0
- 跳过：0
- 错误：0
- 总耗时：4.94 s

项目文档曾记录生成环境下为 27 passed、1 skipped；本机已安装 MuJoCo，因此相关运行时测试实际执行，本次结果为 28 passed。

## PINCH / WRAP 接触指标

### PINCH 任务 CSV

源文件：`outputs/run_20260914_120322.csv`  
归档副本：`PINCH_run_20260914_120322.csv`

| 分析区段 | frames | contact_proxy | stable_proxy | mean_hand_contacts | mean_tip_ratio | servo_mean_error | servo_max_error |
|---|---:|---:|---:|---:|---:|---:|---:|
| PINCH | 231 | **0.0%** | **0.0%** | 0.00 | 0.00 | **0.0647 m** | **0.1163 m** |
| WRAP（同一输入序列内） | 318 | 0.0% | 0.0% | 0.00 | 0.00 | — | — |

判定：PINCH 失败。项目验证目标要求 thumb/index 指尖同时接触且 `pinch_contact_proxy > 0`，本次 PINCH 区段代理率为 0.0%。运行日志各采样点也显示 `contacts=0`。

### WRAP 任务 CSV

源文件：`outputs/run_20260914_120517.csv`  
归档副本：`WRAP_run_20260914_120517.csv`

| 分析区段 | frames | contact_proxy | stable_proxy | mean_hand_contacts | mean_tip_ratio | servo_mean_error | servo_max_error |
|---|---:|---:|---:|---:|---:|---:|---:|
| PINCH（同一输入序列内） | 231 | 0.0% | 0.0% | 1.35 | 0.09 | — | — |
| WRAP | 318 | **41.5%** | **41.5%** | 1.61 | 0.28 | **0.0154 m** | **0.0727 m** |

判定：WRAP 通过当前代理目标；`wrap_contact_proxy > 0`，且稳定接触代理率同为 41.5%。

## compare_mapping.py 主要结果

| 指标 | static | dynamic | improvement |
|---|---:|---:|---:|
| PINCH position MSE | 0.00369089 | 0.00090223 | +75.56% |
| PINCH vector MSE | 0.00331327 | 0.00087813 | +73.50% |
| WRAP coupling penalty | 0.11345632 | 0.01874627 | +83.48% |
| WRAP vector MSE | 0.00430635 | 0.00467090 | **-8.47%** |

该脚本自身注明结果属于 mock-data engineering checks，不是项目报告或论文的最终实验结果。

## 发现的问题

1. **PINCH 接触失败（阻断整体通过）**：PINCH 区段 `contact_proxy=0.0%`、`stable_proxy=0.0%`、`mean_hand_contacts=0.00`、`mean_tip_ratio=0.00`；伺服平均/最大误差分别为 0.0647 m / 0.1163 m。对应日志：`04_mujoco_pinch_1200.log`、`06_analyze_pinch.log`，原始数据：`PINCH_run_20260914_120322.csv`。
2. **WRAP vector MSE 退化**：动态映射相对静态映射为 -8.47%，虽然 coupling penalty 明显改善。对应日志：`03_compare_mapping.log`。
3. **控制台字符编码问题**：运行日志中 `[DONE]` 后的中文路径提示在 PowerShell 5.1 捕获时出现乱码；命令、退出码、CSV 文件名和测试指标不受影响。对应日志：`02_sim_none_900.log`、`04_mujoco_pinch_1200.log`、`05_mujoco_wrap_1200.log`。

## 最终结论

**v0.3.4 未通过当前阶段自动测试。**

理由：基础自动测试（28/28）、无后端运行、映射对比脚本和 WRAP 接触测试均完成，所有测试命令退出码均为 0；但当前阶段明确包含 PINCH 接触验证，而本次 PINCH 的 `contact_proxy` 与 `stable_proxy` 均为 0.0%，未达到项目文档要求的 `pinch_contact_proxy > 0`。因此整体结论为不通过，建议后续优先检查 PINCH 目标可达性、任务物体位置/碰撞几何、接触辅助与 task-space servo 增益。
