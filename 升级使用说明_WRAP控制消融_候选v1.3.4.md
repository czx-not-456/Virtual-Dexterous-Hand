# 从 v1.3.3 候选版升级到 WRAP 控制消融候选 v1.3.4

## 升级方法

如果你已经在 Windows 本地运行 v1.3.3 项目，请优先使用“覆盖补丁 ZIP”，**把压缩包中 `src/`、`configs/`、`experiments/`、`test/`、`docs/`、`README.md` 等直接覆盖到原项目根目录**；不要复制到根目录下的第二层子文件夹，也不需要重新安装 Python。建议覆盖前备份原项目。

你原来的路径示例：

```text
D:\Vhand\Virtual-Dexterous-Hand_接触增强与消融优化_候选v1.3.3_完整项目\Virtual-Dexterous-Hand-contact-v1.3.3-candidate
```

如果新下载了完整项目压缩包，也可独立解压到新目录并重新创建 `.venv`，不要把旧 `.venv` 整个复制过去。

## 最快验证

```powershell
python -m pytest -q test/tests_20261012/test_wrap_servo_policy_ablation.py
python experiments/benchmark_wrap_servo_policy.py --trials 0 --steps 800
```

将自动运行四次 MuJoCo：`original`、`low_gain`、`thumb_recovery`、`coordinated`。数据保存在 `outputs/wrap_servo_policy_ablation/`，查看 `report.md`、`per_trial.csv`、`aggregate.csv`、`manifest.json`。

若需要八个合成试次：

```powershell
python experiments/benchmark_wrap_servo_policy.py --trials 0,1,2,3,4,5,6,7 --steps 800 --output-dir outputs/wrap_servo_policy_8trials
```

若要直观判断圆柱是否被推走又复位，应在四种方法**全部使用相同设置**的前提下运行：

```powershell
python experiments/benchmark_wrap_servo_policy.py --trials 0 --no-auto-reset-object --no-inactive-object-reset --output-dir outputs/wrap_servo_no_teleport
```

研究背景、消融公平性、指标解释与局限见：`docs/validation/WRAP_SERVO_ABLATION_V134.md`。

## 注意

- `--wrap-servo-policy` 默认为 `original`，因此原有命令仍使用旧控制行为。
- 新策略**仅支持** `--task wrap --mapping-method legacy --sim mujoco`；不支持某些六维耦合伺服路径，程序会直接拒绝，不会假装执行了消融。
- 使用原有的接触评价判据，不放宽 1 秒、1 mm、1° 稳定阈值。
- 目前尚没有在本交付环境中获得四种新策略的真实 MuJoCo 对比结果；候选版不能宣传为已证明改进。
