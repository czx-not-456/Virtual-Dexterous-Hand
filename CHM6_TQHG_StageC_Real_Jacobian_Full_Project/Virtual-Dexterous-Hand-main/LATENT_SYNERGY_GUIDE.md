# CH-M6 数据手套 15/6→低维→11 协同映射（可选实验版本）

此分支在不修改第三方 `CH-M6/CH-M6_L.xml` 的前提下，新增一个可选的低维协同映射实验。默认行为保持原状：**只有明确传入 `--latent-model` 才开启新映射**。

## 1. 先确认真实存在的驱动关系

- `数据手套数据/base_mapping(1).json` 是 **15 个输入槽位、6 个输出控制量** 的回归表达式（表达式中的 `&4` 指第 4 个传感器；在 NumPy 中对应索引 3）。事实上该映射标记使用 11 个传感器编号，并非每个通道都参与原 6 路回归。
- `数据手套数据/mod/joint_mapping.json` 中 `CH-M6` 的映射模式为 `shared_6d_then_qn_coupling`：
  - 6 个主动**软件控制量**依次为 `th_cmc_joint, th_mcp_joint, ff_mcp_joint, mf_mcp_joint, rf_mcp_joint, lf_mcp_joint`。
  - `th_ip_joint = 1.5483870968 × th_mcp_joint`
  - `ff_dip_joint = 0.9617834395 × ff_mcp_joint`
  - `mf_dip_joint = 0.9808917197 × mf_mcp_joint`
  - `rf_dip_joint = 0.9617834395 × rf_mcp_joint`
  - `lf_dip_joint = 0.9808917197 × lf_mcp_joint`
- 以上角度在耦合前均以 **degrees** 表示，转换为 radians 后再按 `configs/robot_hand.yaml` 限幅。
- `CH-M6/CH-M6_L.xml` 包含 **11 个 MuJoCo position actuator**。**不能**将软件耦合比解释成真机必有的电机/腱耦合。真实驱动关系必须通过机械资料或实测标定确认。

## 2. 数学方法

定义：传感器 `s ∈ R^15`，原 6 维控制指令 `u ∈ R^6`，共享潜变量 `z ∈ R^k`，11 个关节目标 `q ∈ R^11`。

1. 使用标定数据中的成对观测 `(s_i, u_i)`。按 6 维控制上限将 `u` 归一化到 `[0, 1]`。
2. PCA 提取 `P ∈ R^(6×k)`：`z_u = (u_norm - mean_u) P`。默认 `k=4`，**不预设所有实时手势的本征维数就是 4**。
3. 对输入手套数据进行各通道的 min-max 归一化，并以岭回归学习 `z_s = s_norm W + b`，使其逼近第 2 步定义的共享潜变量。
4. 运行时可选 `sensor`、`control` 或 `blend` 模式：`z=z_s`、`z=z_u` 或两者加权平均（默认更推荐 `sensor`，因为原 base_mapping 可能存在静态偏移）。
5. 还原 6 控制量 `û=clip(mean_u+z P^T,0,1)×control_limits_deg`；通过第 1 节中的既有 Qn 软件耦合关系，扩展为 11 维 `q`。
6. 使用 `IntentDrivenRetargeter` 直接在 **k 维 `z` 上**做 SLSQP 优化，优化器的每一次目标函数评估都会把 `z` 解码为遵守耦合关系的 11 关节姿态。最终接触控制和滤波之后，再投影到协同空间并沿低维轨迹进行关节速度限制，防止下游模块破坏协同关系。

训练误差与独立手势/实物闭环性能不同，尤其对于未被训练数据覆盖的单指独立运动，这种低维表示会有信息损失。建议比较 `k=3/4/5/6`，在新增手势采集数据之后根据独立测试集误差和接触成功率选维度。

## 3. 运行命令

在项目根目录执行（需已安装依赖，特别是 `numpy`, `scipy`, `PyYAML`）：

```bash
# 已提供一个同设备序列号左手标定记录训练的 4 维参数文件
python -m src.main --input glove --sim none --steps 24 \
  --latent-model configs/latent_chm6_k4.json \
  --latent-source sensor --output-dir outputs --run-name demo_latent

# 自行训练（示例：同一个序列号 G12LC040704 的另一批 190 帧左手数据）
python scripts/train_latent_synergy.py \
  --samples '数据手套数据/records/chm6_calibrations/20260917_102742_G12LC040704_left/captured_samples.csv' \
  --out configs/latent_chm6_k4.json --k 4

# 恢复原始映射基线，不带 --latent-model 即可
python -m src.main --input glove --sim none --steps 24

# 单元测试
python -m pytest test/tests_20261012/test_latent_synergy.py -q
```

项目中的 `数据手套数据/captured_samples(1).xlsx`、`base_mapping(1).json`、`manifest(1).json` 可以被已有 `resolve_delivery_path()` 自动识别，不需要改名。如果替换设备，请更新映射模型及对应标定数据。

## 4. 当前实验指标与已知限制

在 **左手序列号 G12LC040704** 的 `20260917_102742` 采集记录上，共使用 190 帧、6 类预设姿态的目标角度标签，得到：

- `k=4` 时，PCA 保留归一化 6 维标签样本方差约 **99.711%**；第 5 个主成分仍存在，因此 4 维不是精确无损映射。
- 同姿态组内末段 20% 帧作诊断性留出时，各输出角度平均绝对误差依次约为 `[2.69°, 0.70°, 0.50°, 0.83°, 0.39°, 0.39°]`。该验证**有相同姿态出现在训练与留出集**，不能视作对未知动作的泛化性能证明。
- 在更晚的另一批 `captured_samples(1).xlsx` 的 24 帧上，6 输出的跨批次平均绝对误差约为 **3.44°**（分轴 `[5.17°, 2.65°, 4.00°, 5.01°, 1.91°, 1.91°]`）；握拳姿态的平均误差约为 7.38°。这仍是对预设控制标签的偏差，不是实际机器人角度误差或抓取任务成功率。
- 低维映射模块的 5 项专用单元测试和 `--sim none` 的端到端 24 帧运行已通过。本环境未安装 `mujoco`，完整真实接触仿真尚未验证。

**下一步需要采集的数据**：至少张开、五指分别屈曲/伸展、拇指对掌、食指/中指/无名指/小指独立动作、二指/三指捏合、握拳、抓握及动态过渡；逐一记录 15 路数据、6 控制目标、11 关节的真实观测值（如果可取得）、手指接触结果及时间戳。对不同日期与不同受试者/手套采用独立验证划分，若 4 维显著损害独立手指控制，请提高 `k` 至 5 或 6，或使用意图条件化的多分支协同表示。
