# Synthetic MuJoCo Benchmark Report

- Dataset: F:\大三上\大创\项目文件夹\Virtual-Dexterous-Hand-v0\datasets\synthetic_glove_v1.csv
- Frames per run: complete selected trial
- Command/summary failures: 0
- Completed runs that failed their required task metric: 0
- Contact success and stable success are reported separately.
- These are fixed-base proxies, not lift/transport success metrics.

| Task | Trials | Required pass | Contact success | Stable success | Contact ratio | Stable ratio | P95 latency (ms) | Joint RMSE (rad) | Tip ratio | Max drift (deg) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| neutral | 3 | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 123.06 | 0.2197 | 0.000 | 0.00 |
| pinch | 3 | 100.0% | 100.0% | 0.0% | 99.2% | 3.5% | 116.21 | 0.2032 | 0.398 | 12.27 |
| wrap | 3 | 100.0% | 100.0% | 100.0% | 93.0% | 9.8% | 119.36 | 0.3412 | 0.638 | 5.84 |
