# v0.3.2 — task contact closed-loop prototype

本版本重点把“动作像抓握”推进到“真实发生手-物接触”。

## 主要变化

1. 保留 v0.3.1 五指模型，并合并侧置、向内屈曲的大拇指修复。
2. 新增 `TaskContactAssist`：
   - PINCH：只对拇指/食指逐步增加闭合量；某指接触物体后停止继续闭合该指。
   - WRAP：五指独立闭合；每根手指一旦接触物体就停止额外闭合。
   - 离开目标意图后自动平滑释放辅助量。
3. 修正接触统计语义：`contact_count` 现在只统计手-物接触；物体-桌面/地面的接触改记为 `environment_contact_count`。
4. 新增逐指接触和指腹接触指标：`thumb_tip_contact`、`index_tip_contact` 等。
5. 新增任务代理指标：
   - `pinch_contact_proxy = thumb_pad AND index_pad`
   - `wrap_contact_proxy = thumb contact AND any non-thumb finger contact`
6. 微调 PINCH 薄块和 WRAP 圆柱的初始位置/尺寸，使其更落在当前五指可达区域。
7. 新增 `experiments/analyze_contacts.py`，可对一次运行日志统计 PINCH/WRAP 接触代理成功比例。

## 注意

这些辅助闭合量和物体初始位姿是当前虚拟原型的工程参数，不是文献或真实 CasiaHand 的官方标定值。接入真实数据手套和真实机械参数后需要重新标定。
