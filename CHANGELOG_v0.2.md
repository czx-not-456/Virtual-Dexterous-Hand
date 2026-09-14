# v0.2 Changelog

- 升级 15-DoF MuJoCo 仿人手外观：掌面/掌根/鱼际/腕部/五指分节。
- 增加拇指、食指及其余手指的可视化指尖标记。
- 修正 MJCF 屈曲轴方向，使正屈曲角与 Python 运动学一致。
- 新增抓取台、动态圆柱测试物体和 PINCH 观察点。
- 新增 overview / closeup / side 固定相机和 free 自由视角。
- 新增 `configs/simulation.yaml`。
- MuJoCo 驱动兼容 Windows 中文路径。
- 新增 `--steps 0` 持续运行和“关闭 Viewer 即退出”。
- 新增测试物体越界自动复位。
- 新增 3 项 MJCF 合同测试及 1 项可选 MuJoCo schema 运行时测试。
