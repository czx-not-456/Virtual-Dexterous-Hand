# 项目目录整理记录（2026-11-07）

按用途区分系统测试与研究实验。现有 experiments/ 未遍历审查，迁入前检查目标不存在，没有覆盖原文件。首次测试收集曾通过旧测试导入其中的兼容入口并报错；随后将该测试改为直接导入新位置，未编辑原入口。

## 迁移清单

逐项来源、去向见 [manifest.json](manifest.json)，共 37 项文件或目录迁移：

| 来源 | 去向 |
| --- | --- |
| 根目录实验01～10 | experiments/ 下同名目录 |
| scripts/analyze_expert_complementarity.py | experiments/StageA_专家互补性/；scripts/ 保留兼容入口 |
| scripts/evaluate_tqhg_v2_faults.py | experiments/实验01_StageB七方法消融/；scripts/ 保留兼容入口 |
| 实验总导航 | experiments/映射对比实验导航.md |
| 实验整理说明、旧迁移 CSV | experiments/整理记录/ |
| SCHEMES/SCHEME_B_VALIDATION.json | 实验05的 results/ |
| SCHEMES/ 的设计文档 | docs/design/SCHEMES/ |
| 通用指南、设计依据、验证说明、变更历史 | docs/guides/、design/、validation/、history/ |
| tqhg_online_patch/ | archive/patches/tqhg_online_patch/；修正安装器定位，不运行安装器 |
| 申报书.pdf、成果梳理_版本1.3.0.md | 项目进度报告相关文件/ |
| outputs/ 下 glove_smoke、glove_tasks_smoke、readme_check、test_logs | test/outputs/ 下同名目录 |

项目根路径、实验模块导入、根目录与 scripts/ 兼容入口、Markdown 链接和当前文档运行示例已作相应调整。配置和仿真默认模型路径指向用户已有的新模型位置，未移动或改写模型资源。

## 删除清单

删除前逐文件比对 SHA256，保留文件路径和摘要见 manifest.json。共删除 7 个重复结果文件：

- outputs/glove_calibration_check/ 的 mapping_comparison.json、mapping_comparison_by_pose.csv。
- test/test_results_v100_20261008/glove_calibration/ 的上述两个文件。
- test/test_results_v100_20261008/glove_tasks/ 的 frames.csv、summary.csv、summary.json。

前四个文件保留在实验03的 results/；后三个保留在实验04的 results/。历史测试报告中的对应链接已更新。没有删除独立目录 README、兼容入口、补丁 payload 或软件依赖。

## 保留与待定清单

- test/ 中的系统测试、JUnit、回归输出、审计记录和混合报告保留；没有整批移动 test_results_*。
- outputs/ 的 run_*、manual，以及未确认用途或内容不同的结果保留。缺少可靠的用途依据时不把它们归作研究实验，也不按同名删除。
- 数据手套数据/ 的原始数据、软件运行库、设备配置、标定快照和模型保持原位；datasets/ 保留共享数据。
- src/、configs/ 及共享训练、标定、数据生成、交付审计工具保持共享位置，实验引用使用。
- archive/older_root_versions/ 和完整补丁 payload 保留；部分补丁文件虽与主代码相同，但删除会破坏补丁完整性。
- 根目录 Stage C 说明是文档跳转入口，几个审计脚本和 PowerShell 是兼容入口，继续保留。
- 用户原有模型迁移、验证文档变动及原有 experiments/ 内容未恢复、覆盖或整理。原有 experiments/ 内入口未逐项验证，后续优先使用新实验导航和已验证的 scripts/ 入口。
- 历史报告、CSV 迁移清单中的旧命令、数据哈希和当时状态作为历史记录保留；当前去向以本次 manifest.json 为准。

## 验证结果

- 系统测试：136 passed，耗时 181.23 秒；[JUnit](../../test/organization_20261107/pytest.xml)、[运行日志](../../test/organization_20261107/pytest.log)。
- 9 个保留入口的 --help，以及 Stage C 单条件、单方法 --dry-run：全部成功；没有执行完整研究实验。详见 [入口记录](../../test/organization_20261107/entry_checks.json)。
- 本次迁入 Python 文件和保留入口的 AST 语法检查通过；7 个去重保留文件的 SHA256 复核通过。
- 根目录和实验10的 PowerShell 脚本语法检查通过，保留 UTF-8 BOM 以兼容 Windows PowerShell 中文解析；[记录](../../test/organization_20261107/powershell_checks.json)。

未安装依赖、未运行手套软件、未重训模型、未执行稳定性比较或其他完整研究实验，未提交 Git。
