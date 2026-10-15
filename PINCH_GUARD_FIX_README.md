# CH-M6 / TEQS 捏合回退修复 — 仿真试验补丁

## 故障定位

在持续捏合数据回放中 `PINCH=238` 且 `spectrum_post_guard=238`，经源码核实，该字段表示**最终护栏拒绝候选动作并回退为6维可行参考**。该批次对 `PINCH` 的拒绝率为100%。原先 `ObjectAwareTaskServo` 在11个独立关节上提出修正，而 TEQS 的最终护栏在6维耦合空间中要求指令与数据手套参考相差不超过约1.5°，并比较的是人手对应的任务目标（不是蓝色物体的接触目标）。

## 修改内容

1. `task_space_servo.py`：新增 `apply_coupled`，将拇指和食指的世界坐标误差通过 `J_q C` 合并求解，直接输出6维耦合控制的11关节指令。
2. `mujoco_env.py`：新增 `fingertip_positions_for_target`，使用独立 MuJoCo 数据对象做正向运动学，不改动正在运行的仿真状态。
3. `task_effect_spectrum.py`：捏合伺服激活时，对候选命令与速度约束下的参考命令计算**到当前物体表面接触目标的预测指尖误差**；仅允许物体误差不明显变差且参考偏移未超限的候选。非接触或无 MuJoCo 模式依旧采用原有手套姿态护栏。
4. `main.py`：连接新的伺服器和物体任务护栏，并在 CSV 输出三个诊断列：`spectrum_guard_reason`, `spectrum_object_error_m`, `spectrum_object_baseline_m`。
5. `configs/task_effect_spectrum.yaml`：新增仿真专用护栏参数，默认允许比原来1.5°更大的**任务修正偏离**（60°上限），但6→11耦合、关节角限位和单帧速度限制仍然保持。

**重要：** 物体接触误差是对目标**关节姿态**的运动学预测，不是实测执行后的真实接触误差，也不是碰撞安全保证。修改是故障定位和仿真试验版本，尚未在本环境运行完整 MuJoCo 捏合实验。

## Windows 使用

在项目根目录先备份：

```powershell
Copy-Item src\main.py src\main.py.bak
Copy-Item src\retargeting\task_effect_spectrum.py src\retargeting\task_effect_spectrum.py.bak
Copy-Item src\retargeting\task_space_servo.py src\retargeting\task_space_servo.py.bak
Copy-Item src\simulation\mujoco_env.py src\simulation\mujoco_env.py.bak
Copy-Item configs\task_effect_spectrum.yaml configs\task_effect_spectrum.yaml.bak
```

解压本 ZIP 文件到 `Virtual-Dexterous-Hand-main` **根目录**，允许合并并覆盖同名文件。也可在该目录执行：

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\CHM6_TEQS_Pinch_Guard_Fix.zip" -DestinationPath . -Force
```

运行：

```powershell
.\.venv\Scripts\python.exe -m src.main --input glove --glove-samples "数据手套数据\CHM6_pinch_hold_diagnostic_only.xlsx" --sim mujoco --render --realtime --camera closeup --task pinch --steps 360 --run-name pinch_guardfix_test --latent-model configs/latent_chm6_adaptive.json --task-effect-spectrum --spectrum-config configs/task_effect_spectrum.yaml --latent-source sensor
```

然后查看护栏触发原因：

```powershell
$rows = Import-Csv "outputs\pinch_guardfix_test.csv"
$pinch = @($rows | Where-Object { $_.intent -eq 'PINCH' })
"PINCH帧数：$($pinch.Count)"
$pinch | Group-Object spectrum_guard_reason | Select-Object Name,Count
$pinch | Measure-Object -Property spectrum_post_guard -Average
```

原因码：`0` 接受，`1` 任务误差变差，`2` 超出物体控制参考偏移上限，`3` 运动学预测失败而安全回退。

观察 `[RESULT]` 里的 `contact_success`, `stable_success`, `contact_ratio`；并观察 `Terr`, `Ierr` 是否随捏合逐步下降。

## 测试状态

在本环境执行非 MuJoCo 相关测试：

```text
15 passed
```

纯算法 `--sim none` 回放24帧成功；本环境尚无 mujoco 包，无法验证真实接触闭环是否成功。

## 回滚

出现不正常运动立即停止仿真。可把上述 `.bak` 文件复制回原名称进行回滚。此补丁仅供仿真验证，未经真机安全评估。
