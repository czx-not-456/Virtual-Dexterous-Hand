# 中期检查增强成果 V2

本目录在原项目基础上增加独立离线分析和展示材料，未改动原映射、控制和仿真模块。建议先打开本目录的成果图册，再按需查看「结果」中的高清图、GIF 与数据表。

## 成果目录

| 编号 | 已完成内容 | 证据性质 |
|---|---|---|
| 05 | 捏合与包络抓握的 CH-M6 三维关键帧 | 原 MJCF/STL + 历史 11 关节记录的运动学重建；不是新 MuJoCo 截图 |
| 06 | 两种历史姿态 GIF | 离线关节姿态回放；不含物体、碰撞及接触力 |
| 07 | 系统流程图 | 现有工程与新增验证关系 |
| 08 | 两种任务时序数据曲线 | 原有 MuJoCo 日志 trial 00 的重分析 |
| 09 | NEUTRAL/PINCH/WRAP 共9次试验 | 原有试验统计；代理指标不是实体抓取成功率 |
| 10 | Stage B 七方法缺失条件消融 | 既有配对实验重绘；非本次重新训练 |
| 11 | 测试记录与下一阶段计划 | 本次执行的检查 + 既有工程无参数测试 |
| 13 | 六类标定姿态识别及缺失鲁棒性 | 本次真实采集记录离线实验，按设备留出 |
| 14 | EMA 滤波参数压力测试 | 本次合成输入噪声注入实验，不是物理测量 |

## Windows 运行

在项目根目录执行（Python 3.12 可用；已经创建了虚拟环境就不要重复创建）。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r "中期增强成果V2\requirements-v2.txt"
.\.venv\Scripts\python.exe "中期增强成果V2\复现实验.py"
.\.venv\Scripts\python.exe "中期增强成果V2\验证成果.py"
.\.venv\Scripts\python.exe "中期增强成果V2\生成图册.py"
```

离线脚本默认重绘全部图，三维回放可能耗时一分钟以上；仅统计可加 `--skip-3d`。
数据按相对项目根目录读取，生成结果保存在本目录的「结果」。

### 现场 MuJoCo 实时展示（需要本机 MuJoCo）

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& ".\中期增强成果V2\实时演示.ps1" -Task pinch
& ".\中期增强成果V2\实时演示.ps1" -Task wrap
```

也可直接运行原项目命令：

```powershell
.\.venv\Scripts\python.exe -m src.main --sim mujoco --render --realtime --input dataset --dataset-trial 0 --task pinch --steps 800 --camera closeup
.\.venv\Scripts\python.exe -m src.main --sim mujoco --render --realtime --input dataset --dataset-trial 0 --task wrap --steps 800 --camera overview
```

本次离线环境没有 MuJoCo，未重新运行物理仿真；本机首次执行需确认依赖、窗口和接触效果。真实截图请在 Viewer 中截取。不要将 05、06 的离线图称为新物理实验截图。

## 新增实验方法

13：使用 captured_samples.csv 中六个共有类别，排除派生训练样本和额外类别，按设备序列号留一设备法。每次仅用其他设备拟合标准化与等设备权重的类别原型，再对留出设备预测。缺失通道数量 0、1、3、5，每个非零条件以固定随机种子生成10组掩码，所有方法共享同一掩码；对比训练均值填补与忽略缺失维度的平均平方距离。汇总以设备等权，另有逐帧混淆矩阵（样本等权）。这是静态标定标签的识别验证，不代表连续动态手势或新被试泛化。

14：使用原有8条合成输入序列作为参考，每通道注入标准差为参考序列标准差12%的独立零均值高斯噪声；比较因果 EMA α=1、0.7、0.45、0.25 的归一化 RMSE 和帧间差分 RMS。原始合成输入不是无噪声物理真值，也不据此声称实体跟踪精度提升。

## 建议中期现场顺序

1. 说明输入、映射、虚拟手及评价框架（07）。
2. 展示两个姿态重建 GIF，随后运行实际 MuJoCo（05、06）。
3. 用时序与九次结果讲清接触代理和稳定代理的区别（08、09）。
4. 展示既有 Stage B 消融与新增跨设备识别、缺失压力测试（10、13）。
5. 说明滤波在平滑性和偏差间的权衡（14）。
6. 强调捏合稳定性仍待提高，下一阶段开展动态动作、随机物体位姿和真实持续持物验证（11）。
