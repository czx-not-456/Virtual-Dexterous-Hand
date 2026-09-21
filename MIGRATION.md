# MIGRATION

本文档汇总项目的历史版本迁移记录，用于说明各版本之间的模型、配置、代码与验证方式变化，不作为当前版本的安装指南。

## v0.1 → v0.2

v0.2 不改变上层“数据手套 → 标定/滤波 → 姿态解算 → 意图识别 → 动态映射”的总体算法链路，重点升级 MuJoCo 展示与交互验证层。

### 主要变化

1. 新增 `models/dexterous_hand/humanoid_hand_v02.xml`
   - 仍保持 15 个关节和原有 actuator 名称；
   - 手掌、腕部、五指采用更接近人手比例的组合几何；
   - 拇指与食指指尖采用醒目的不同标记，便于观察 PINCH；
   - 其余三指指尖统一标记，便于观察 WRAP；
   - 关节屈曲轴改为 `-X`，与 Python 前向运动学“正角度向 -Z 屈曲”的符号约定一致。

2. 重做抓取场景
   - 增加抓取台；
   - 将测试物体替换为中央圆柱体，并赋予 `freejoint`，支持物理接触和后续抓取实验；
   - 增加半透明捏合观察点；
   - 测试物体离开工作区后可自动复位，适合长时间演示。

3. 新增相机
   - `overview`：默认斜前方总览；
   - `closeup`：近距离观察手指接触；
   - `side`：侧面观察屈曲轨迹；
   - `free`：不绑定固定相机，手动调整视角。

4. 修复 Windows 中文路径
   - 对包含非 ASCII 字符的工程路径，用 Python 先读取 MJCF 文本，再调用 `MjModel.from_xml_string()`。

5. 支持持续运行
   - `--steps 0` 表示无限运行；
   - 关闭 MuJoCo Viewer 或在终端按 `Ctrl+C` 可结束。

### 推荐启动命令

```powershell
python -m src.main --sim mujoco --render --steps 0 --realtime
```

近景：

```powershell
python -m src.main --sim mujoco --render --steps 0 --realtime --camera closeup
```

自由视角：

```powershell
python -m src.main --sim mujoco --render --steps 0 --realtime --camera free
```

## v0.3 → v0.3.1

需要覆盖/新增以下文件：

- `models/dexterous_hand/humanoid_hand_v031.xml`（新增并作为默认模型）
- `models/dexterous_hand/simple_hand.xml`
- `configs/simulation.yaml`
- `src/simulation/mujoco_env.py`
- `src/main.py`
- `tests/test_mjcf_model.py`
- `CHANGELOG_v0.md` 中的 `CHANGELOG_v0.3.1` 小节

升级后先运行：

```powershell
pytest -rA
```

然后建议：

```powershell
python -m src.main --sim mujoco --render --steps 0 --realtime --task wrap --camera overview
```

`overview` 现在只是自由相机的初始角度，不再锁定视角。

## v0.4.0 → v0.4.1

*PINCH contact fix*

This patch keeps the v0.4.0 synthetic dataset and the successful WRAP setup, and targets the remaining precision-pinch failure.

### What changed

1. `task_space_servo.pinch.stop_on_contact: tip`
   - non-tip phalanx contact no longer freezes thumb/index before their pads touch.
2. `target_offset_m: -0.002`
   - uses a small virtual penetration target. MuJoCo collision stops the real pad at the object surface.
3. Stronger pinch preshape (`posture_prior_blend=0.90`, `command_blend=0.90`).
4. PINCH object widened in X while staying thin in Y, reducing toppling without making the pinch gap easier in Y.
5. Terminal prints `Ttip`, `Itip`, `Terr`, and `Ierr` every 30 active frames.
6. Keeps the WRAP-success servo tuning: gain 0.88, max step 6.5 deg, stop error 0.8 mm, wrap clearance 5.5 mm.

### Run

```powershell
python -m src.main --sim mujoco --render --realtime --input dataset --dataset datasets\synthetic_glove_v1.csv --dataset-trial 0 --task pinch --steps 600 --camera closeup
```

Success target:

```text
Ttip=1 Itip=1
[RESULT] task=pinch success_proxy=True ...
```

`success_proxy` remains the fixed-base stable-contact proxy; it is not lift/transport success.
