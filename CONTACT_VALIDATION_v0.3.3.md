# v0.3.3 接触验证步骤

1. `pytest -rA`
2. PINCH：
   `python -m src.main --sim mujoco --render --steps 0 --realtime --task pinch --camera closeup`
3. WRAP：
   `python -m src.main --sim mujoco --render --steps 0 --realtime --task wrap --camera overview`
4. 对两次最新 CSV 分别执行：
   `python experiments/analyze_contacts.py outputs\\run_XXXXXXXX_XXXXXX.csv`

## 当前最低目标

PINCH：
- `thumb_tip_contact = True`
- `index_tip_contact = True`
- `pinch_contact_proxy = True`

WRAP：
- `thumb_contact = True`
- 至少一根非拇指 `*_contact = True`
- `wrap_contact_proxy = True`

若仍无法达标，下一步才进入 object-aware task-space servo，而不是继续只靠固定关节角增量。
