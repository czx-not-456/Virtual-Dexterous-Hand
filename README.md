# 映射方法核心消融（精简候选版）

已将重复/低证据的实验从主清单移出；主线为 **Stage B 鲁棒性、B/C/D 模块消融、方案 E 效率/维度消融，Stage C 待验证**。移出的 02、03、04、10 均在 `archive/低证据实验审计/` 保留可审计历史，不以删数据制造更好结果。

- [核心实验导航](experiments/映射对比实验导航.md)
- [一键运行与证据报告](experiments/run_core_mapping_suite.py)
- [筛选说明与局限](docs/validation/MAPPING_CORE_SELECTION.md)

```powershell
python experiments/run_core_mapping_suite.py --dry-run
python experiments/run_core_mapping_suite.py --groups stageb,modules,spectrum --steps 24 --out outputs/mapping_core
```

## 方案 E（任务效应谱）重点研究

本项目将方案 E 作为重点研究方向之一，明确区分动态选维、固定维度、快速求解器和最终非线性守卫的贡献。研究说明与当前结果见 [方案 E 研究深化计划与当前结论](experiments/实验08_任务效应谱方案对比/方案E研究深化计划与当前结论.md)。当前 24 帧/6 姿态探索性重跑证据见 [方案 E 深化消融结果](experiments/实验08_任务效应谱方案对比/results/generated/方案E深化消融_24帧_2次重复/report.md)。

推荐运行：

```powershell
python experiments/benchmark_scheme_e_research.py --dry-run
python experiments/benchmark_scheme_e_research.py --steps 24 --repeats 3 --out outputs/scheme_e_research
```

v1.3.6 候选版已完成 E1–E5 的代码与评估流程推进：E1 增加候选拒绝原因并修正多条件接受证书；E2 将谱预算、编码器保真、非线性 FK/regret、速度/关节可行性联合纳入选维；E3 增加同一 E 求解链路下的动态 k 与固定 k=4/5/6 对照；E4 完成 27 份跨采集记录的 E_dynamic vs E_fixed6 回放（12 个手套序列号/左右手组合）；E5 增加 MuJoCo PINCH/WRAP 及 Pipeline P95/P99、deadline miss、任务空间误差评估入口。

跨记录结果显示 E_dynamic 的实际低秩输出比例为 95.66%、fallback 为 0%，MAE 与固定 k=6 基本持平（记录级均值差约 +0.011°），但 Pipeline P95 平均慢约 0.52 ms。因此当前证据支持“降低输出秩且基本保持离线标签误差”，不支持“精度更高”或“计算更快”。E4 使用预设姿态标签且无独立受试者 ID，不能称为独立受试者泛化。E5 的代码与 dry-run 已验证，实际 MuJoCo 物理运行需在安装 MuJoCo 的环境执行。

详细结果见 [方案 E E1–E5 完整推进报告](docs/research/方案E_E1-E5完整推进报告_v1.3.6.md) 和 [Windows 运行说明](方案E_E1-E5运行说明_v1.3.6.md)。可用 `experiments/run_scheme_e_e1_e5.ps1` 一次运行 E1–E4 并检查 E5 命令；在有 MuJoCo 的机器上追加 `-RunMujoco` 执行 E5，追加 `-Render` 可打开实时可视化。

---

> **新增可选研究方案 B：任务自适应协同 + 受限残差。** 运行方法、公式、实验记录和局限见 [`SCHEME_B_ADAPTIVE_GUIDE.md`](docs/design/SCHEMES/SCHEME_B_ADAPTIVE_GUIDE.md)。默认入口仍按原工程执行，启用时需传 `--latent-model configs/latent_chm6_adaptive.json --adaptive-synergy`。

## 项目介绍：Virtual-Dexterous-Hand CH-M6

支持合成手套数据与真实手套标定数据回放、意图驱动重定向和 MuJoCo 仿真的虚拟灵巧手实验系统。

当前版本提供：

- 指导老师提供的 CH-M6_L 模型（11 DoF / 11 position actuator）；
- NEUTRAL、PINCH、WRAP 意图识别；合成数据另包含 OPEN 过渡阶段；
- 数据集回放和程序化 Mock 输入；
- 指导老师提供的真实数据手套标定样本、base mapping、留出样本个性化映射对照和 `sim`/`glove` 输入切换；
- neutral、PINCH、WRAP 三类标准场景；
- 基于物体位姿、指尖 Jacobian 和欠驱动可达方向的任务空间伺服；
- 每帧 CSV、任务 summary JSON、批量 CSV、Markdown 报告和 HTML 看板；
- 明确分离的接触成功代理与稳定成功代理。

项目统一使用 `CH-M6/` 中指导老师提供的只读第三方模型；site、接触代理和任务场景均在运行时注入，不修改原始 XML/STL。当前手掌基座固定，因此所有成功指标都是接触代理，不代表完成抓起、搬运或放置。

## 1. Windows 新电脑从零部署

### 1.1 前置条件

- Windows 10 或 Windows 11，64 位；
- 推荐 Python 3.11；
- 支持范围为 Python 3.10–3.12；
- 建议使用 PowerShell 5.1 或 PowerShell 7；
- MuJoCo 由 Python 包安装，不需要另外下载可执行程序。

确认 Python：

    py -3.11 --version

如果命令不存在，请先从 Python 官方安装 Python 3.11，并在安装器中启用 Python Launcher。

### 1.2 进入项目并创建虚拟环境

    cd "你的路径\Virtual-Dexterous-Hand-v0"
    py -3.11 -m venv .venv

PowerShell 当前进程若禁止脚本，可临时放开：

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

激活环境：

    .\.venv\Scripts\Activate.ps1

升级 pip 并安装依赖：

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip check

requirements.txt 包含 NumPy、SciPy、PyYAML、MuJoCo 和 pytest。

### 1.3 PowerShell UTF-8 设置

主程序和 benchmark 已主动以 UTF-8 输出。为保证 PowerShell 5.1 的重定向、管道和第三方命令也统一使用 UTF-8，建议在当前终端执行：

    $env:PYTHONUTF8 = "1"
    [Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
    $OutputEncoding = [Console]::OutputEncoding

这些设置只影响当前 PowerShell 窗口。

## 2. 快速验证安装

运行完整测试：

    python -m pytest -rA

当前版本包含纯算法、配置、MuJoCo schema、可达性以及 neutral/PINCH/WRAP canonical trial 集成测试。安装了 MuJoCo 的标准环境应执行全部测试，不应跳过物理运行测试。

测试结果以本机实际运行输出为准；其中 MuJoCo 集成测试需要已安装的物理后端。

检查依赖：

    python -m pip check

运行映射对照：

    python experiments\compare_mapping.py

2026-10-08 基线下，PINCH 位置 MSE 降低 31.08%，PINCH 向量 MSE 增加 10.24%，WRAP 向量 MSE 降低 0.33%；当前 `synergies: {}`，协同惩罚为零。请按各指标分别解读，不沿用旧模型阶段的欠驱耦合改善结论。

## 3. 输入数据

默认合成数据集：

    datasets\synthetic_glove_v1.csv

数据集包含 8 个 trial，每个 trial 800 帧。完整周期为：

- OPEN：80 帧；
- NEUTRAL：80 帧；
- PINCH：240 帧；
- NEUTRAL：80 帧；
- WRAP：240 帧；
- OPEN：80 帧。

重新生成：

    python scripts\generate_synthetic_dataset.py --trials 8 --frames 800 --seed 42

不要把帧数改为 600 后再用于标准 benchmark，因为那样不会覆盖完整 WRAP 阶段。

`数据手套数据/` 另有指导老师交付的 `captured_samples(1).xlsx`（24 帧、6 种标定姿态）、`base_mapping(1).json` 和 `manifest(1).json`。它们是实采标定数据，与上述 8 个合成 trial 分开使用；程序会兼容省略 `(1)` 的路径写法。

2026-10-12 新增的交付包另含 27 组 CH-M6 标定快照，其中 26 组有对应校正权重，1 组因归一化无效被拒绝；还包含一个初始化权重。标定 CSV 的 `elapsed_ms` 在每个姿态片段重新计时，片段之间没有记录连续过渡。`scripts/audit_teacher_delivery.py` 可只读核对哈希与时间轴；`experiments/replay_teacher_clips.py` 使用其中一组快照按片段内真实时间戳回放基础映射。两者的输出见 `test/test_results_newdata_20261012/`。

    python scripts\audit_teacher_delivery.py --output test\test_results_newdata_20261012\delivery_audit.json
    python experiments\replay_teacher_clips.py --output-dir test\test_results_newdata_20261012\teacher_clips

后一个实验输出基础映射角度误差、由老师的六控制量耦合规则计算的指尖位置误差、片段内输出速度，以及分阶段耗时和 PINCH/WRAP 接触指标。它不加载未知 `.pt`，也不把独立姿态片段解释为自然连续任务。

## 4. 运行单个任务

输入源统一为 `sim` 或 `glove`。模拟源默认回放数据集，也可改为程序化 Mock：

    python -m src.main --sim none --input sim --sim-source dataset --dataset-trial 0 --steps 120
    python -m src.main --sim none --input sim --sim-source mock --steps 120

真实手套标定数据回放使用 `数据手套数据` 中的样本、base mapping 和 manifest。`--input glove` 的主程序运行 base mapping；默认循环回放这 24 帧：

    python -m src.main --sim none --input glove --task pinch --steps 24

在 MuJoCo 中回放相同输入：

    python -m src.main --sim mujoco --input glove --task pinch --steps 240

这两条命令用于检查实采输入和 base mapping 链路；标定姿态顺序不保证产生有效 PINCH 成绩。需要比较两种映射的任务指标时，使用下方的配对实验脚本。

也可通过 `--glove-samples`、`--base-mapping`、`--glove-manifest` 指定其他标定快照。旧的 `--input dataset` 和 `--input mock` 命令仍作为兼容别名保留。

对比 base mapping 与个性化 mapping 在 manifest 声明的每姿态第 4 帧留出集上的误差：

    python experiments\compare_glove_calibration.py

结果写入 `outputs\glove_calibration\mapping_comparison.json` 和 `mapping_comparison_by_pose.csv`。脚本只读取老师提供的模型评估结果，不重新训练或设计映射算法。

使用相同的实采留出姿态输入，在 MuJoCo 中分别运行 base 和 personalized mapping 的 PINCH、WRAP 配对试验：

    python experiments\compare_glove_tasks.py

默认每组 240 帧，结果写入 `outputs\glove_tasks\summary.csv`、`summary.json` 和 `frames.csv`；运行参数、指标定义和实验限制见 `experiments/实验04_手套抓握任务对比/GLOVE_TASK_COMPARISON.md`。该旧实验仍使用旧 manifest 的第 4 帧预测并重复回放姿态，不加载新增权重，不能据此声称已完成连续在线推理。

### 4.1 PINCH

无界面、完整 canonical trial：

    python -m src.main --sim mujoco --input dataset --dataset datasets\synthetic_glove_v1.csv --dataset-trial 0 --task pinch --steps 800

可视化并按 60 Hz 实时运行：

    python -m src.main --sim mujoco --render --realtime --input dataset --dataset datasets\synthetic_glove_v1.csv --dataset-trial 0 --task pinch --steps 800 --camera closeup

CH-M6 PINCH 方块沿 X 轴由拇指/食指对向夹持，实际尺寸为 16 × 50 × 120 mm，底面位于 0.070 m 桌面上。

### 4.2 WRAP

无界面：

    python -m src.main --sim mujoco --input dataset --dataset datasets\synthetic_glove_v1.csv --dataset-trial 0 --task wrap --steps 800

可视化：

    python -m src.main --sim mujoco --render --realtime --input dataset --dataset datasets\synthetic_glove_v1.csv --dataset-trial 0 --task wrap --steps 800 --camera overview

### 4.3 纯算法模式

不启动 MuJoCo：

    python -m src.main --sim none --input dataset --dataset-trial 0 --task pinch --steps 800

此模式用于检查数据、标定、意图、映射、日志链路；因为没有物理后端，接触指标应为零。

### 4.4 Viewer 操作

可用相机预设：

- overview；
- closeup；
- side；
- free。

预设仅设置自由相机初始位置，启动后仍可用 MuJoCo Viewer 鼠标旋转、平移和缩放。

## 5. 三场景批量 benchmark

运行全部三类任务（neutral、pinch、wrap）、trial 0：

    python experiments\benchmark_synthetic.py --trials 1

运行全部三类任务、前 5 个 trial：

    python experiments\benchmark_synthetic.py --trials 5

只运行 PINCH 和 WRAP：

    python experiments\benchmark_synthetic.py --trials 1 --tasks pinch wrap

默认不需要传 --steps。脚本会读取每个所选 trial 的实际行数并完整运行。仅在明确需要截断诊断时才使用，例如：

    python experiments\benchmark_synthetic.py --trials 1 --tasks pinch --steps 400

批处理会分别统计：

- command_failures：子进程非零退出、summary 缺失或 summary 无效；
- task_metric_failures：命令完成，但任务要求的接触或稳定指标未通过。

command_failures 大于零时 benchmark 自身返回非零退出码。任务指标失败会进入报告，但不会被伪装成命令执行失败。

输出目录默认为：

    outputs\benchmark

包含：

- benchmark_runs.csv：每个 task/trial 的完整结果，含 returncode 和 error；
- benchmark_summary.csv：按任务聚合的接触成功率、稳定成功率和任务通过率；
- BENCHMARK_REPORT.md：可阅读报告；
- benchmark_dashboard.html：HTML 看板；
- 每次子运行的 CSV 和 summary JSON。

## 6. 输出文件

单次运行默认写入 outputs：

    run_YYYYMMDD_HHMMSS_task.csv
    run_YYYYMMDD_HHMMSS_task_summary.json

可以指定目录和名称：

    python -m src.main --sim mujoco --task wrap --steps 800 --output-dir outputs\manual --run-name wrap_trial0

逐帧 CSV 主要字段：

- intent：识别出的手势意图；
- contact_count：手与任务物体的接触对数量；
- environment_contact_count：物体与桌面/地面的接触数量；
- thumb_contact 等：逐指任意部位接触；
- thumb_tip_contact 等：逐指指腹接触；
- task_contact_proxy：当前任务需要的瞬时接触；
- stable_contact_proxy：任务接触且物体姿态漂移不超过阈值；
- contact_success_proxy：任务接触连续达到配置帧数后锁存为真；
- stable_success_proxy：稳定接触连续达到配置帧数后锁存为真；
- contact_success_streak_frames：当前连续任务接触帧数；
- stable_success_streak_frames：当前连续稳定接触帧数；
- orientation_drift_deg：物体相对初始姿态漂移；
- object_displacement_m：物体中心位移；
- servo_mean_error_m：活动手指任务空间平均误差；
- pipeline_latency_ms：单帧完整管线耗时。

summary JSON 同时保留两套成功指标：

- contact_success_proxy：是否建立了规定时长的任务接触；
- stable_success_proxy：是否建立了规定时长的自由体稳定接触；
- required_success_metric：该任务规定使用 contact 还是 stable；
- task_metric_pass：规定指标是否通过。

neutral 要求流程无错误；PINCH 要求 contact；WRAP 要求 stable。接触与稳定两套原始指标始终同时输出。

## 7. 结果分析

分析单次 CSV：

    python experiments\analyze_run.py outputs\run_你的时间戳_task.csv

分析 PINCH/WRAP 接触区段：

    python experiments\analyze_contacts.py outputs\run_你的时间戳_task.csv

映射对照：

    python experiments\compare_mapping.py

实采手套标定误差对照与 MuJoCo 任务对照分别使用 `experiments\compare_glove_calibration.py` 和 `experiments\compare_glove_tasks.py`；后者输出的 `summary.csv` 每个任务与映射一行，`frames.csv` 保存逐帧指标。

生成的 CSV 使用 UTF-8 BOM，便于 Windows Excel 正确识别中文和字段名；JSON、Markdown 与 HTML 使用 UTF-8。

## 8. 主要目录

    configs/       手套、人体手、机器人手、算法和仿真配置
    datasets/      可复现模拟手套数据
    数据手套数据/  指导老师交付的标定样本、映射、权重及便携式软件
    CH-M6/        指导老师提供的 CH-M6 模型与网格
    experiments/   benchmark、实采手套映射/任务对照和结果分析
    scripts/       数据生成和辅助运行脚本
    src/           主程序、驱动、特征、映射、仿真和评测
    test/          按日期归档的测试套件与各版本测试结果
    项目进度简报/  按日期归档的项目进度简报
    outputs/       运行结果；默认被 Git 忽略

当前自动化测试位于 `test/tests_20261012/`，`pyproject.toml` 将 pytest 收集范围限定为该目录，避免扫描历史结果或老师的便携式软件。后续测试套件按 `test/tests_日期编号/` 命名；新结果按 `test/test_results_版本号/测试目的_日期编号/` 归档，并同步更新 pytest 的当前测试目录。既有测试结果保留原归档名称，不改写历史记录。

核心配置：

- configs/simulation.yaml：物体尺寸、位置、任务类型、相机和稳定阈值；
- configs/algorithm.yaml：意图阈值、优化器权重、任务空间伺服和任务专属参数；
- configs/robot_hand.yaml：CH-M6 11 DoF、15→11 映射、运动学链和可配置待确认参数；
- 数据手套数据/CH-M6型号_虚拟手模型/CH-M6_L.xml：指导老师原始只读模型；`src/simulation/ch_m6_adapter.py`：项目侧运行时适配。
- `src/simulation/teacher_chm6_control.py`：可选的老师版 6 维命令到 11 个仿真关节的独立适配层；不修改原始 MJCF 或默认 11 actuator 配置。

## 9. 当前边界与已知问题

- 已接入真实手套标定数据并完成留出姿态的 MuJoCo 回放；合成数据 benchmark 与这组实采实验应分别解读；
- 新增包包含 26 个与有效快照配对的个性化权重，但缺少可核验的 PWL/LoRA 前向推理实现；本工程未反序列化或运行这些权重。旧任务对照依然只使用 manifest 预测，不是连续在线推理；
- 手掌基座固定，没有手腕/机械臂抓起和搬运阶段；
- 结构、摩擦、腱路和关节参数仍是工程近似；
- PINCH 的接触成功可能包含接近夹具阶段，稳定成功只统计释放后的自由体帧；
- 新增标定记录是分段姿态采集，各段有时间戳，段间过渡未被采集；不能当成自然连续操作或跨用户任务试验；
- 正式论文若要论证真实抓取或跨用户效果，仍需要真实硬件、更多受试者和标准物体重复试验。

版本演进与设计依据见 CHANGELOG_v0.md、LITERATURE_INFORMED_DESIGN.md 和 VALIDATION.md。


## 目录整理（2026-11-07）

- 系统测试、报告和输出：`test/`。
- 研究对比与消融：[实验导航](experiments/映射对比实验导航.md)。
- 通用文档：`docs/`；项目材料：`项目进度报告相关文件/`。
- 迁移、去重与保留说明：[整理记录](docs/organization/README.md)。

## WRAP 接触后协调控制四方法消融（候选 v1.3.4）

> 新增研究候选策略尚未获得 MuJoCo 性能验证；原始方案保持默认。

在项目根目录运行 `python experiments/benchmark_wrap_servo_policy.py --trials 0`，即可用相同数据、相同场景和相同物理判据执行 `original`、`low_gain`、`thumb_recovery`、`coordinated` 四组 WRAP 对照。批量运行和严格评估说明见 [`docs/validation/WRAP_SERVO_ABLATION_V134.md`](docs/validation/WRAP_SERVO_ABLATION_V134.md)。
