# 接触判定稳健性消融（与 Stage C 物理抓握评估分开）

本消融比较 4 个确实不同的判断器：

- `geometry_only`：原始几何候选接触，无力过滤/时序门控。
- `force_only`：只检验法向力阈值，无时间滤波。
- `temporal_only`：保留弱接触，只有时间滤波。
- `force_and_temporal`：法向力阈值 + 力迟滞 + 连续帧进入/退出（当前改进）。

在仓库根目录运行：

```powershell
python experiments/benchmark_contact_detection.py --episodes 40 --seed 20261108
python -m pytest -q test/tests_20261012/test_contact_detection_robustness.py
```

输出为同一合成接触序列上的配对 F1、误检率、切换次数、检测延迟和按 episode bootstrap 的探索性 CI。
这些 episode 是**人工生成的信号**，没有 MuJoCo 真实动力学，也不代表不同被试/独立采集数据。结果只说明某种噪声模型下的判断器行为，不代表抓握成功率。

**下一步必须做的物理验证**：固定种子和配对场景，对干净/单通道缺失/多通道缺失及多种物体、摩擦、扰动场景，分别关闭/开启力门控、迟滞、时间门控，记录同一真值参考下的误判、接触阶段延迟、检测切换率、滑移、自由物体 1s 相对位姿稳定率。避免将接触滤波后的保持时间当作稳定性真值。

## 可实际运行的 MuJoCo 控制器消融

四个检测器都已通过 `--contact-detector` 接入主程序与 Stage C 驱动；**控制器看到的接触状态**因方法而异，但 `task_contact_proxy`、稳定代理始终由相同的本帧物理力证据计算，避免改变评价尺子。使用同一个输入、物体和故障掩码，在仓库根目录依次运行：

```powershell
python scripts/run_stageC_mujoco.py --contact-detector geometry_only --outdir outputs/contact_geometry
python scripts/run_stageC_mujoco.py --contact-detector force_only --outdir outputs/contact_force
python scripts/run_stageC_mujoco.py --contact-detector temporal_only --outdir outputs/contact_temporal
python scripts/run_stageC_mujoco.py --contact-detector force_and_temporal --outdir outputs/contact_combined
```

当前包内包含 `数据手套数据/CHM6_pinch_hold_diagnostic_only.xlsx` 诊断工作簿。若替换为独立采集的新连续动作输入，可用 `--glove` 指定。每个 detector 单独一个 `--outdir`，防止覆盖。请重复不同**独立输入片段**与摩擦/物体条件，并以跨片段统计而非全帧伪重复做结论；当前发布包没有本组 MuJoCo 结果。

已附带 `results/published_synthetic_example/`：包含 40 个**合成** episode 的示例 CSV/JSON，不应与真实 MuJoCo 输出混淆。
