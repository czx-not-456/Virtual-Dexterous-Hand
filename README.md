# Virtual-Dexterous-Hand v0.4.6

基于模拟数据手套、意图驱动重定向和 MuJoCo 的虚拟灵巧手实验系统。

当前版本提供：

- 15 DoF / 7 actuator 欠驱动手模型；
- OPEN、NEUTRAL、PINCH、WRAP 意图识别；
- 数据集回放和程序化 Mock 输入；
- pinch、wrap、sphere、card、bottle、box 六类标准任务；
- 基于物体位姿、指尖 Jacobian 和欠驱动可达方向的任务空间伺服；
- 每帧 CSV、任务 summary JSON、批量 CSV、Markdown 报告和 HTML 看板；
- 明确分离的接触成功代理与稳定成功代理。

项目使用工程化近似手模型，不是 CasiaHand 官方 CAD 数字孪生。当前手掌基座固定，因此所有成功指标都是接触代理，不代表完成抓起、搬运或放置。

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

当前版本包含纯算法、配置、MuJoCo schema、可达性以及六任务 canonical trial 集成测试。安装了 MuJoCo 的标准环境应执行全部测试，不应跳过物理运行测试。

当前仓库在 Python 3.11.9、MuJoCo 3.13.0 环境的基准结果为 58 passed。

检查依赖：

    python -m pip check

运行映射对照：

    python experiments\compare_mapping.py

WRAP vector MSE 当前存在约 -8.47% 的已知多目标权衡；动态映射同时显著降低欠驱动 coupling penalty。测试设置了退化上限，防止该指标继续静默恶化。

## 3. 数据集

默认数据集：

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

## 4. 运行单个任务

### 4.1 PINCH

无界面、完整 canonical trial：

    python -m src.main --sim mujoco --input dataset --dataset datasets\synthetic_glove_v1.csv --dataset-trial 0 --task pinch --steps 800

可视化并按 60 Hz 实时运行：

    python -m src.main --sim mujoco --render --realtime --input dataset --dataset datasets\synthetic_glove_v1.csv --dataset-trial 0 --task pinch --steps 800 --camera closeup

PINCH 方块实际尺寸为 90 × 30 × 140 mm。MuJoCo box 配置使用半尺寸 [0.045, 0.015, 0.070]，中心高度为 0.140 m，底面位于 0.070 m 桌面上。

### 4.2 WRAP

无界面：

    python -m src.main --sim mujoco --input dataset --dataset datasets\synthetic_glove_v1.csv --dataset-trial 0 --task wrap --steps 800

可视化：

    python -m src.main --sim mujoco --render --realtime --input dataset --dataset datasets\synthetic_glove_v1.csv --dataset-trial 0 --task wrap --steps 800 --camera overview

### 4.3 其他标准物体

    python -m src.main --sim mujoco --input dataset --dataset-trial 0 --task sphere --steps 800
    python -m src.main --sim mujoco --input dataset --dataset-trial 0 --task card --steps 800
    python -m src.main --sim mujoco --input dataset --dataset-trial 0 --task bottle --steps 800
    python -m src.main --sim mujoco --input dataset --dataset-trial 0 --task box --steps 800

card 是 6 mm 薄边、30 mm 夹持宽度、140 mm 高度的校准卡片。PINCH/card 接近阶段会临时固定物体：PINCH 首次建立双指接触后释放，薄 card 连续建立规定帧数的真实双指接触后释放。夹具期间可计入接触成功，但不会计入稳定成功。

### 4.4 纯算法模式

不启动 MuJoCo：

    python -m src.main --sim none --input dataset --dataset-trial 0 --task pinch --steps 800

此模式用于检查数据、标定、意图、映射、日志链路；因为没有物理后端，接触指标应为零。

### 4.5 Viewer 操作

可用相机预设：

- overview；
- closeup；
- side；
- free。

示例：

    python -m src.main --sim mujoco --render --realtime --task sphere --steps 800 --camera side

预设仅设置自由相机初始位置，启动后仍可用 MuJoCo Viewer 鼠标旋转、平移和缩放。

## 5. 六任务批量 benchmark

运行所有六类任务、trial 0：

    python experiments\benchmark_synthetic.py --trials 1

运行所有六类任务、前 5 个 trial：

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

PINCH 和 card 要求 contact；wrap、sphere、bottle、box 要求 stable。两套原始指标始终同时输出。

## 7. 结果分析

分析单次 CSV：

    python experiments\analyze_run.py outputs\run_你的时间戳_task.csv

分析 PINCH/WRAP 接触区段：

    python experiments\analyze_contacts.py outputs\run_你的时间戳_task.csv

映射对照：

    python experiments\compare_mapping.py

生成的 CSV 使用 UTF-8 BOM，便于 Windows Excel 正确识别中文和字段名；JSON、Markdown 与 HTML 使用 UTF-8。

## 8. 主要目录

    configs/       手套、人体手、机器人手、算法和仿真配置
    datasets/      可复现模拟手套数据
    experiments/   benchmark、映射对照和结果分析
    models/        MuJoCo MJCF 手与任务场景
    scripts/       数据生成和辅助运行脚本
    src/           主程序、驱动、特征、映射、仿真和评测
    tests/         单元、回归和六任务 MuJoCo 集成测试
    outputs/       运行结果；默认被 Git 忽略

核心配置：

- configs/simulation.yaml：物体尺寸、位置、任务类型、相机和稳定阈值；
- configs/algorithm.yaml：意图阈值、优化器权重、任务空间伺服和任务专属参数；
- configs/robot_hand.yaml：15 DoF、7 actuator、关节范围和欠驱动耦合；
- models/dexterous_hand/humanoid_hand_v031.xml：当前 MuJoCo 模型。

## 9. 当前边界与已知问题

- 当前使用模拟数据，不等于真实数据手套实验；
- 手掌基座固定，没有手腕/机械臂抓起和搬运阶段；
- 结构、摩擦、腱路和关节参数仍是工程近似；
- PINCH/card 的接触成功可能包含接近夹具阶段，稳定成功只统计释放后的自由体帧；
- WRAP vector MSE 相对静态映射约退化 8.47%，作为欠驱动 coupling 改善的已知多目标权衡保留；
- 正式论文实验仍需要真实硬件、真实受试者/手套标定和标准物体重复试验。

版本演进与设计依据见 CHANGELOG_v0.md、LITERATURE_INFORMED_DESIGN.md 和 VALIDATION.md。
