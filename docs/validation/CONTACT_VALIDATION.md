# CONTACT_VALIDATION

本文档汇总 v0.3.2 至 v0.3.4 的接触验证流程。各节记录对应历史版本的验证方法与判定标准，其中测试数量等结果仅代表当时版本，不代表当前项目状态。

## v0.3.2

建议按以下顺序测试：

```powershell
pytest -rA
python -m src.main --sim mujoco --render --steps 0 --realtime --task pinch --camera closeup
python -m src.main --sim mujoco --render --steps 0 --realtime --task wrap --camera overview
```

### PINCH 目标

当终端进入 `intent=PINCH` 后，观察：
- 拇指从掌侧向内收拢；
- 食指向薄块另一侧靠近；
- 日志中 `thumb_tip_contact=True` 与 `index_tip_contact=True` 能同时出现；
- `pinch_contact_proxy=True` 至少在 PINCH 时段的一部分帧出现。

### WRAP 目标

当进入 `intent=WRAP` 后，观察：
- 拇指从圆柱侧面参与包络，而不是离开圆柱；
- 至少一根非拇指与圆柱接触；
- `thumb_contact=True` 且 `finger_contact=True`；
- `wrap_contact_proxy=True` 能持续若干帧。

### 日志分析

程序退出后执行：

```powershell
python experiments/analyze_contacts.py outputs/run_YYYYMMDD_HHMMSS.csv
```

该统计只是当前虚拟原型的接触代理指标，不等同于真实机器人抓取成功率。

## v0.3.3

### 接触验证步骤

1. `pytest -rA`
2. PINCH：
   `python -m src.main --sim mujoco --render --steps 0 --realtime --task pinch --camera closeup`
3. WRAP：
   `python -m src.main --sim mujoco --render --steps 0 --realtime --task wrap --camera overview`
4. 对两次最新 CSV 分别执行：
   `python experiments/analyze_contacts.py outputs\\run_XXXXXXXX_XXXXXX.csv`

### 当前最低目标

PINCH：
- `thumb_tip_contact = True`
- `index_tip_contact = True`
- `pinch_contact_proxy = True`

WRAP：
- `thumb_contact = True`
- 至少一根非拇指 `*_contact = True`
- `wrap_contact_proxy = True`

若仍无法达标，下一步才进入 object-aware task-space servo，而不是继续只靠固定关节角增量。

## v0.3.4

### 1. 自动测试

```powershell
pytest -rA
```

本生成环境结果：27 passed, 1 skipped。跳过项为可选 MuJoCo runtime 测试；本机已安装 MuJoCo 时应执行该项。

### 2. PINCH

```powershell
python -m src.main --sim mujoco --render --steps 0 --realtime --task pinch --camera closeup
```

目标：thumb_tip_contact 与 index_tip_contact 同时出现，pinch_contact_proxy > 0。

### 3. WRAP

```powershell
python -m src.main --sim mujoco --render --steps 0 --realtime --task wrap --camera overview
```

目标：thumb_contact 与至少一个非拇指 contact 同时出现；圆柱在 WRAP 开始前保持标准直立初态。

### 4. 统计

```powershell
python experiments/analyze_contacts.py outputs\run_YYYYMMDD_HHMMSS.csv
```

除 contact_proxy 外，v0.3.4 会输出 servo_mean_error 与 servo_max_error。若接触率仍低，可据误差判断是目标不可达、伺服增益不足，还是发生了错误部位接触。
