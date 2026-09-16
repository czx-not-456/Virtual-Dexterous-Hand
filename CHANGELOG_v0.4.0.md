# CHANGELOG v0.4.0

## 目标

在不接真实数据手套的前提下，将 v0.3.4 推进为可复现实验、可批量评测、可直接形成大创实验结果的数据链路。

## 新增

1. **离线模拟数据集**
   - 新增 `datasets/synthetic_glove_v1.csv`；
   - 8 个 trial，每个 trial 600 帧；
   - 包含 OPEN / NEUTRAL / PINCH / WRAP；
   - 每个 trial 加入通道增益、偏置、噪声与手掌姿态扰动，用于模拟不同佩戴/个体差异。

2. **CSV 回放驱动**
   - `src/glove/dataset_driver.py`；
   - 支持 `--input dataset --dataset-trial N`；
   - 其余姿态解算、意图识别、映射、MuJoCo 控制代码无需区分输入来自实时设备还是离线数据。

3. **标准测试物体目录**
   - `wrap` 圆柱；
   - `pinch` 薄块；
   - `sphere` 球体；
   - `card` 卡片；
   - `bottle` 瓶状圆柱；
   - `box` 长方体。

4. **PINCH 接触可达性修正**
   - box 捏合目标保留当前指尖 X 坐标，避免要求固定无外展自由度的手指强制对准中心线；
   - 将 PINCH 垂直边界余量从 10 mm 调整到 1 mm；
   - 增加与当前手几何一致的拇指/食指预定位先验；
   - 提高任务空间闭环主导比例和单帧最大修正量。

5. **统一任务成功代理与评测**
   - 新增 `TaskEpisodeEvaluator`；
   - 连续稳定接触达到配置帧数后判定 `success_proxy=True`；
   - 自动统计接触率、稳定率、首次接触/成功时间、P95 延迟、RMSE、速度 RMS、姿态漂移等。

6. **自动批量实验与结果看板**
   - `experiments/benchmark_synthetic.py`；
   - 生成逐次实验 CSV、聚合 CSV、Markdown 报告、HTML 看板。

## 兼容性

- 保留原 `MockGloveDriver`；使用 `--input mock` 可回到程序生成式 mock；
- 默认输入改为 `dataset`，以保证实验可重复；
- 原 `pinch / wrap / sphere` 命令仍可用。

## 验证状态

当前生成环境完成：

- Python 静态编译通过；
- 全部非 MuJoCo runtime 测试通过；
- dataset 回放 600 帧纯算法 smoke run 通过；
- 批处理脚本在 `--sim none` 下流程通过。

当前容器没有 MuJoCo 包，因此 **PINCH/WRAP 的真实物理接触改善必须在已安装 MuJoCo 的 Windows 机器上复测**。不要把未复测的 `success_proxy` 写成最终实验结论。
