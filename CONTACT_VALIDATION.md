# v0.3.2 contact validation

建议按以下顺序测试：

```powershell
pytest -rA
python -m src.main --sim mujoco --render --steps 0 --realtime --task pinch --camera closeup
python -m src.main --sim mujoco --render --steps 0 --realtime --task wrap --camera overview
```

## PINCH 目标

当终端进入 `intent=PINCH` 后，观察：
- 拇指从掌侧向内收拢；
- 食指向薄块另一侧靠近；
- 日志中 `thumb_tip_contact=True` 与 `index_tip_contact=True` 能同时出现；
- `pinch_contact_proxy=True` 至少在 PINCH 时段的一部分帧出现。

## WRAP 目标

当进入 `intent=WRAP` 后，观察：
- 拇指从圆柱侧面参与包络，而不是离开圆柱；
- 至少一根非拇指与圆柱接触；
- `thumb_contact=True` 且 `finger_contact=True`；
- `wrap_contact_proxy=True` 能持续若干帧。

## 日志分析

程序退出后执行：

```powershell
python experiments/analyze_contacts.py outputs/run_YYYYMMDD_HHMMSS.csv
```

该统计只是当前虚拟原型的接触代理指标，不等同于真实机器人抓取成功率。
