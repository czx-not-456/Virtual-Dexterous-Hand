# v0.4.0 本机验证步骤

在项目根目录执行：

```powershell
.\.venv\Scripts\Activate.ps1
pytest -rA
```

然后依次执行：

```powershell
python -m src.main --sim mujoco --input dataset --dataset-trial 0 --task pinch --steps 600
python -m src.main --sim mujoco --input dataset --dataset-trial 0 --task wrap --steps 600
python -m src.main --sim mujoco --input dataset --dataset-trial 0 --task sphere --steps 600
```

优先检查 `*_summary.json`：

- `success_proxy`
- `contact_frame_ratio`
- `stable_frame_ratio`
- `max_success_streak_frames`
- `first_contact_latency_ms`
- `p95_pipeline_latency_ms`
- `max_orientation_drift_deg`
- `mean_servo_error_m`

PINCH 目标：相较 v0.3.4 的 `pinch_contact_proxy=0`，至少应首先出现 thumb/index 同时指腹接触。若仍为 0，保留 CSV，不要继续凭观察调参，优先根据 `servo_thumb_error_m` / `servo_index_error_m` 与接触位置定位问题。

批量实验：

```powershell
python experiments\benchmark_synthetic.py --trials 5 --steps 600
```

最终用于大创阶段汇报的表格建议直接取 `outputs/benchmark/benchmark_summary.csv`，可视化演示使用 `benchmark_dashboard.html`。
