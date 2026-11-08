# 方案 B：任务自适应协同 + 受约束残差（可选研究原型）

> **状态：已实现并通过纯算法测试；尚无 MuJoCo 接触验证、真机标定或独立受试者评测。**
> 原始 `base_mapping(1).json` 的 15→6 映射和 `joint_mapping.json` 的 6→11 耦合依旧保留，原始运行模式不受影响。所有角度除已标注为 rad 外均为 deg。

## 1. 代码与开关

- 新增 `src/retargeting/adaptive_synergy.py`：任务相关雅可比维度选择、有界互补空间残差、六控制量到 11 关节的约束映射、滤波后的约束投影和限速。
- 扩展 `src/retargeting/latent_synergy.py` 的训练输出（版本 2）：保存完整 6×6 PCA 基，以及 15→6 维传感器岭回归预测器；**原先的 k 维模型仍可照常使用**。
- 修改 `src/retargeting/optimizer.py`：方案 B 用最多 6 个 PCA 系数优化，不再优化 11 个彼此独立的关节，并施加六控制量限位、关节限位、残差幅值、正则化等约束。
- 修改 `src/main.py` 与 `src/glove/adapter.py`：提供 `--adaptive-synergy` 开关，完整传递 15 个传感器和6维控制参考；后处理对投影后的6维控制命令限速，保留受限残差。日志增加 `adaptive_k`、`adaptive_projection_error_m`、`adaptive_residual_norm` 和 `adaptive_feasible`。
- 参数见 `configs/adaptive_synergy.yaml`；预训练研究参数见 `configs/latent_chm6_adaptive.json`。

## 2. 算法（适用于论文方法部分的候选定义）

令传感器为 `s∈R15`、软件控制为 `u∈R6`、CH-M6 关节指令为 `q∈R11`。设归一化控制 `v=u/L`，PCA 全基 `B∈R6×6`，均值 `μ∈R6`，其中 L 是六维角度范围。

**双编码器：**由成对标定数据学习传感器控制回归 `v_s=R(s)`，并保留已有的 k 维编码器 `z_s`；原 `base_mapping.json` 提供 `v_c`。运行时 `sensor`、`control`、`blend` 选择或混合两路参考，得到 `v_ref`。请注意这只是对六类控制标签的回归，不是独立测量的机器人目标角度。

**按任务挑选维度：**在前一可行指令 `v_prev` 处，针对意图 `c∈{NEUTRAL,PINCH,WRAP}` 的关键指尖以有限差分求取任务空间雅可比 `J_c∈R^(3m×6)`。设任务增强映射

```
A_c = [ J_c ; λ_c I6 ]
d   = A_c (v_ref - v_prev)
E_k = || d - A_c B_k (A_c B_k)^† d ||2
k*  = min { k ∈ [k_min,k_max] : E_k <= max(ε_abs, ε_rel ||d||2) }
```

若均不满足，选 `k_max`。λ_c 让独立手指小位移也能被注意到；PINCH 优先拇指/食指，WRAP 考虑全部五指。**并没有声称训练了任务专属 PCA 基**：这里是同一个全局 PCA 基，在不同任务雅可比下自适应截断；任务与 PCA 的条件化可作为未来升级方向。

**受限互补残差：**用完整的六维协同基 `B=[B_k,N_k]`，解码

```
v = μ + B_k z + N_k δ
q = C v
```

`C∈R11×6` 中已经包含六控制通道范围（度→弧度）和五组软件耦合比。优化目标使用现有指尖位置/相对向量/拓扑/平滑/碰撞软罚项，再加 `λ_ref mean((v-v_ref)^2) + λ_res ||δ||2^2`；**约束：**

```
v_lower <= v <= v_upper
-δ_max(intent) <= δ <= δ_max(intent)
q_lower <= C v <= q_upper  （已通过预计算 v_lower, v_upper 实现）
```

这里的 `δ` 是互补方向的**有界优化系数**，并不意味着 CH-M6 存在 11 个独立控制自由度。六个软件控制量是当前配置的有效驱动空间；真实传动关系尚需标定。指尖自碰撞沿用原工程**软惩罚**，不是接触碰撞“硬约束”。

**后处理：**为防止物体接触伺服/滤波后的 11 维修改破坏6→11耦合，先把输出投影至当帧 `k,δ_max` 对应的可行控制集合，再沿六维命令线段按每个关节速度限制插值。若任务切换造成残差幅值约束瞬时改变，先满足连续速度，再逐步过渡到新的残差幅值约束；不能同时假定这两者在每次切换的首帧都严格成立。后处理使用上一帧**指令**而不是可能因接触/惯性偏离耦合流形的实测位置。

## 3. 运行与测试

从工程根目录执行：

```bash
pip install -r requirements.txt

# 旧方法基线：不加 adaptive 参数
python -m src.main --input glove --sim none --steps 24 \
  --latent-model configs/latent_chm6_adaptive.json --latent-source sensor

# 新方法 B（完整6×6基）
python -m src.main --input glove --sim none --steps 24 \
  --latent-model configs/latent_chm6_adaptive.json \
  --adaptive-synergy --adaptive-config configs/adaptive_synergy.yaml \
  --latent-source sensor --output-dir outputs --run-name scheme_b_24

# 相同输入、相同命令链路的固定PCA与B诊断对照
python experiments/实验05_PCA与自适应协同对比/benchmark_adaptive_synergy.py --steps 24 --source sensor

# 算法单元测试（不需 MuJoCo）
python -m pytest test/tests_20261012/test_adaptive_synergy.py -q

# 所有无需 MuJoCo 的现有+新增测试
python -m pytest -k 'not standard_tasks_integration' -q
```

若要重训（不要混合左右手/设备序列号）：

```bash
python scripts/train_latent_synergy.py \
  --samples '数据手套数据/records/chm6_calibrations/20260917_102742_G12LC040704_left/captured_samples.csv' \
  --out configs/latent_chm6_adaptive.json --k 4
```

调参入口：`k_min,k_max` 控制主协同维数；`projection_abs_m,projection_relative` 控制维度门限；`command_scale_m` 控制任务中独立控制维度的重视程度；`residual_caps` 控制互补残差最大幅值（归一化系数，不是关节角）；`lambda_residual,lambda_reference` 控制任务拟合与偏离主协同之间的权衡。所有默认值是**工程实验初值**，尚未对独立抓握数据调优。

## 4. 当前跑通的证据及局限

- 190帧标定数据中只有**6种目标向量**，中心化秩 **5**。4维 PCA 对这些离散标签解释约 **99.71%** 方差，并不能说明“真实手势只有4维”。第六 PCA 方向来自零方差补空间，**缺乏真实独立指令训练覆盖**。
- 一批独立的24帧上传数据全部被当前意图识别器判成 `NEUTRAL`，所以这批重放不能证明动态 PINCH/WRAP 抓取效果。新增单元测试构造 `PINCH` 和 `WRAP` 场景检查数学行为，并非真实抓握。
- 在纯算法模式、同24帧传感器输入下，本次诊断固定 PCA k=4 的6控制预设标签 MAE 约 **20.85°**、B 约 **20.33°**；B 的动态主维度分布为 k=2(13帧)、k=3(5帧)、k=4(4帧)、k=5(2帧)；两者均无关节越界与明显耦合误差。**这些较大的控制误差受起始状态、低通滤波、每帧6°速度限制、有限帧轨迹等因素影响，不是离线回归准确率；0.52° 的变化不能当成统计上可靠的提升。**
- 该环境没有 `mujoco`，因此没有验证与物体接触成功率、稳定性、摩擦条件或真机性能。计算耗时只是本机离线纯算法测量，不能作为实时控制截止期保证。见 `SCHEME_B_VALIDATION.json`。

## 5. 作为论文实验还需要做什么

1. 补采连续动作：五指独立屈伸、二指/三指捏合、圆柱、球体抓握和缓慢切换。记录 15 路传感器原始角、真实6维控制参考、11维实际关节与接触状态。训练/测试按采集批次、手套和受试者隔离。
2. 在同环境和同一输入下对照：原始15→6→11；固定k=3/4/5 PCA；完整6维控制+约束；方案B去掉动态k；方案B去掉残差；完整B。
3. 每种操作重复多次并报告：指尖误差、抓取成功率、接触持续、碰撞/限位违反、轨迹抖动、计算时间p95，以及失效率和置信区间。
4. 特别验证：低维阶段减少的计算负担是否被任务雅可比和受限优化抵消；若没有降低实时成本，就应把贡献定位为**同等控制成本下精细动作保持/可控性**，而非无依据地强调加速。
