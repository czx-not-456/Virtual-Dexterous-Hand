# 从 v0.3 升级到 v0.3.1

需要覆盖/新增以下文件：

- `models/dexterous_hand/humanoid_hand_v031.xml`（新增并作为默认模型）
- `models/dexterous_hand/simple_hand.xml`
- `configs/simulation.yaml`
- `src/simulation/mujoco_env.py`
- `src/main.py`
- `tests/test_mjcf_model.py`
- `CHANGELOG_v0.3.1.md`

升级后先运行：

```powershell
pytest -rA
```

然后建议：

```powershell
python -m src.main --sim mujoco --render --steps 0 --realtime --task wrap --camera overview
```

`overview` 现在只是自由相机的初始角度，不再锁定视角。
