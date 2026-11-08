# 历史测试去重说明（2026-11-08）

本目录原有 23 个测试源码。19 个文件与当前套件逐内容比较一致，仅 LF/CRLF 换行不同，已删除旧副本；对应关系见下表。

| 删除的旧文件 | 保留的当前文件 |
| --- | --- |
| `test_benchmark_synthetic.py` | [test_benchmark_synthetic.py](../tests_20261012/test_benchmark_synthetic.py) |
| `test_calibration.py` | [test_calibration.py](../tests_20261012/test_calibration.py) |
| `test_combined_contact_config.py` | [test_combined_contact_config.py](../tests_20261012/test_combined_contact_config.py) |
| `test_contact_assist.py` | [test_contact_assist.py](../tests_20261012/test_contact_assist.py) |
| `test_dataset_driver.py` | [test_dataset_driver.py](../tests_20261012/test_dataset_driver.py) |
| `test_features_intent.py` | [test_features_intent.py](../tests_20261012/test_features_intent.py) |
| `test_glove_adapter.py` | [test_glove_adapter.py](../tests_20261012/test_glove_adapter.py) |
| `test_mapping.py` | [test_mapping.py](../tests_20261012/test_mapping.py) |
| `test_mjcf_model.py` | [test_mjcf_model.py](../tests_20261012/test_mjcf_model.py) |
| `test_mujoco_runtime.py` | [test_mujoco_runtime.py](../tests_20261012/test_mujoco_runtime.py) |
| `test_physics_clock.py` | [test_physics_clock.py](../tests_20261012/test_physics_clock.py) |
| `test_pinch_fixture_v045.py` | [test_pinch_fixture_v045.py](../tests_20261012/test_pinch_fixture_v045.py) |
| `test_pinch_geometry_v044.py` | [test_pinch_geometry_v044.py](../tests_20261012/test_pinch_geometry_v044.py) |
| `test_pinch_reachability_v046.py` | [test_pinch_reachability_v046.py](../tests_20261012/test_pinch_reachability_v046.py) |
| `test_smoke.py` | [test_smoke.py](../tests_20261012/test_smoke.py) |
| `test_sphere_card_reachability.py` | [test_sphere_card_reachability.py](../tests_20261012/test_sphere_card_reachability.py) |
| `test_task_catalog_xml.py` | [test_task_catalog_xml.py](../tests_20261012/test_task_catalog_xml.py) |
| `test_task_space_servo.py` | [test_task_space_servo.py](../tests_20261012/test_task_space_servo.py) |
| `test_task_workspace.py` | [test_task_workspace.py](../tests_20261012/test_task_workspace.py) |

本目录继续保留以下 4 个历史差异文件，不纳入默认 pytest 收集：

- `test_glove_calibration_experiment.py`：历史实验导入路径。
- `test_robot_architecture.py`：历史 CH-M6 模型路径。
- `test_standard_tasks_integration.py`：历史连续 12 帧稳定成功要求。
- `test_task_evaluator.py`：历史稳定指标语义，与新版 `legacy_*` 及一秒稳定判据不同。

旧文件从 `tests/` 迁入 `test/`，内容保持不变。完整哈希及去重依据见 [迁移清单](../organization_20261108_merge/manifest.json)。
