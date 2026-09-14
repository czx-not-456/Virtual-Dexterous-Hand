# v0.3.1 修复说明

本版本针对 v0.3 实机 MuJoCo 验证中发现的两个可视化问题进行修复。

## 1. 五指模型修复

- 明确保留且仅保留 5 个手指根节点：`thumb1`、`index1`、`middle1`、`ring1`、`little1`。
- 移除容易从正视/斜视角看成“第六根手指”的细长 `forearm` capsule。
- 使用宽矩形 `wrist_base` 表示腕部底座，使腕部与手指在视觉上明显区分。
- 新增自动测试，确保 palm 下不会再出现第六个 digit root。

## 2. 相机改为可拖拽自由视角

v0.3 的 `overview / closeup / side` 会切换为 MuJoCo fixed camera，视点被锁定。
v0.3.1 将它们改为 **FREE camera 初始预设**：

- `overview`：启动时给出整体观察角度；
- `closeup`：启动时靠近拇指/食指与任务物体；
- `side`：启动时提供侧向观察；
- `free`：使用 MuJoCo 默认自由相机。

无论使用前三种哪一种，启动后都仍可通过 MuJoCo Viewer 的鼠标交互全方位观察。

## 3. 同步纳入 v0.3 XML Schema 修复

- `fixed tendon` 不再写入不受支持的 `width` / `rgba` 属性。
- 该修复与你本机 16 项测试全部通过后的版本一致。
