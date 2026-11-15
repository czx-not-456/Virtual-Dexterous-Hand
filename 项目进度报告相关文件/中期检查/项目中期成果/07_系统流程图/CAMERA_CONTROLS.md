# MuJoCo 自由观察说明（v0.3.1）

v0.3.1 的 `overview`、`closeup`、`side` 都只是**启动时的自由相机预设**，启动后不会锁定。

在 MuJoCo Viewer 中可直接使用鼠标进行观察：

- 拖动：旋转/改变观察方向；
- 配合右键或修饰键拖动：平移视点（具体组合随 Viewer/平台版本可能略有差异）；
- 滚轮：拉近/拉远；
- 如不想使用任何预设，可启动 `--camera free`。

推荐命令：

```powershell
python -m src.main --sim mujoco --render --steps 0 --realtime --task wrap --camera overview
```

```powershell
python -m src.main --sim mujoco --render --steps 0 --realtime --task pinch --camera closeup
```
