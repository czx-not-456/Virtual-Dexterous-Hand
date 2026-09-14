# v0.1 → v0.2 迁移说明

v0.2 不改变上层“数据手套 → 标定/滤波 → 姿态解算 → 意图识别 → 动态映射”的总体算法链路，重点升级 MuJoCo 展示与交互验证层。

## 主要变化

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

## 推荐启动命令

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
