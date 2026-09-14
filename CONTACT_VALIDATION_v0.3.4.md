# v0.3.4 接触验证流程

## 1. 自动测试

```powershell
pytest -rA
```

本生成环境结果：27 passed, 1 skipped。跳过项为可选 MuJoCo runtime 测试；本机已安装 MuJoCo 时应执行该项。

## 2. PINCH

```powershell
python -m src.main --sim mujoco --render --steps 0 --realtime --task pinch --camera closeup
```

目标：thumb_tip_contact 与 index_tip_contact 同时出现，pinch_contact_proxy > 0。

## 3. WRAP

```powershell
python -m src.main --sim mujoco --render --steps 0 --realtime --task wrap --camera overview
```

目标：thumb_contact 与至少一个非拇指 contact 同时出现；圆柱在 WRAP 开始前保持标准直立初态。

## 4. 统计

```powershell
python experiments/analyze_contacts.py outputs\run_YYYYMMDD_HHMMSS.csv
```

除 contact_proxy 外，v0.3.4 会输出 servo_mean_error 与 servo_max_error。若接触率仍低，可据误差判断是目标不可达、伺服增益不足，还是发生了错误部位接触。
