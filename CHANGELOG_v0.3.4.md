# v0.3.4 更新说明

本版本基于 v0.3.3 的真实 MuJoCo 截图与两份运行 CSV 做针对性修正。

## 诊断结论

- PINCH 场景：231 个 PINCH 帧中没有发生任何手-物接触，说明固定角度闭合无法修正三维位置误差。
- WRAP 场景：WRAP contact proxy 仅约 9.6%，且圆柱在正式 WRAP 前已被前置动作推倒；稳定抓握代理始终为 0。

## 核心改动

1. 新增 ObjectAwareTaskServo：使用当前任务物体位姿、MuJoCo 指尖位置与 site Jacobian，逐帧把指尖引导到物体表面。
2. 拇指使用 3-DoF damped-least-squares 修正；非拇指沿单腱欠驱动协同方向做 1-DoF 有效 Jacobian 修正。
3. 新增任务门控物体复位：目标意图未激活时保持物体标准初态，进入 PINCH/WRAP 的首帧重新复位后再释放，防止圆柱在前置动作中提前倾倒。
4. 接触后保持对应手指的当前实际姿态，降低持续穿透或把物体推飞的风险。
5. CSV 新增 task-space servo 误差指标；analyze_contacts.py 会同时输出平均/最大任务空间误差。
6. 保留 v0.3.2 固定角度 TaskContactAssist 作为 fallback；当 task_space_servo 开启时不再叠加固定角度辅助。

## 新增 CSV 字段

- task_servo_active
- servo_thumb_error_m
- servo_index_error_m
- servo_mean_error_m
- servo_max_error_m
- servo_corrected_digits

