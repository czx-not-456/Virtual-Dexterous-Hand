# 方案 E E1–E5 运行说明（候选 v1.3.6）

本项目已完成 E1–E4 的离线代码与采集记录评估，增加 E5 MuJoCo 任务/实时验证入口。请在项目根目录打开 PowerShell，并激活项目 `.venv`。

## 1. 环境检查

```powershell
python --version
python -c "import numpy, scipy, yaml; print('core dependencies OK')"
python -c "import mujoco; print('MuJoCo:', mujoco.__version__)"
```

E1–E4 离线流程不要求 MuJoCo；E5 才需要 MuJoCo。

## 2. E1–E3：严格选维诊断与固定维数公平对照

```powershell
python experiments/实验08_任务效应谱方案对比/benchmark_scheme_e_research.py `
  --steps 24 --repeats 3 `
  --modes E_dynamic,E_fixed4,E_fixed5,E_fixed6 `
  --out outputs/e_fair_dimension_controls_user
```

重点查看：
- `report.md`：误差、耗时、低秩输出、fallback 与配对姿态差值；
- `mode_selection.csv`：requested_k、accepted_k、output_rank、候选拒绝原因；
- `paired_pose_deltas.csv`：按姿态配对误差；
- `manifest.json`：输入文件、配置哈希和运行记录。

`--repeats` 是同一批数据的计算重复，只用于测量耗时波动，不增加独立样本量。

## 3. E4：跨采集记录评估（27 份）

```powershell
python experiments/benchmark_scheme_e_sessions.py `
  --modes E_dynamic,E_fixed6 `
  --out outputs/scheme_e_cross_recording `
  --resume
```

支持断点续跑。若某批次运行中断，可再次执行同一命令；完整 session 会被复用。也可用 `--start-index 14 --max-sessions 7` 分批运行；最终汇总会扫描同一输出目录下全部完整 session，不只汇总当前批次。

重点查看 `outputs/scheme_e_cross_recording/report.md`、`method_summary.csv`、`paired_recording_deltas.csv`、`paired_device_hand_deltas.csv` 和 `dimension_summary.csv`。

说明：这些是预设姿态目标标签下的跨采集记录回放。项目没有独立受试者 ID，不能称为独立受试者泛化。

## 4. E5：MuJoCo PINCH/WRAP 与实时验证

先做命令预检查：

```powershell
python experiments/benchmark_scheme_e_mujoco.py --dry-run --tasks pinch,wrap --modes E_dynamic,E_fixed6
```

正式运行：

```powershell
python experiments/benchmark_scheme_e_mujoco.py `
  --tasks pinch,wrap `
  --modes E_dynamic,E_fixed6 `
  --out outputs/scheme_e_mujoco
```

可视化运行可增加 `--render --realtime`。建议先无 GUI 完成指标对照，再单独选择 E_dynamic 运行可视化，以免渲染影响耗时统计。

输出包括 `summary.csv`、`report.md`、每个任务/方法的逐帧 CSV 与 Summary JSON。报告将包含任务空间线性误差、伺服误差、物体位移、Pipeline P95/P99、deadline miss、实际低秩输出和 fallback。

## 5. 解释结果时的限制

- 教师标签误差不是真实机器人关节真值；
- 27 份采集记录不等于 27 名独立受试者；
- 姿态保持序列的命令步长/速度不等于自由手势动态跟踪误差；
- 16.667 ms deadline miss 是软件测量，不是硬实时认证；
- MuJoCo 接触/稳定成功仍是仿真代理，不是硬件抓握证明。
