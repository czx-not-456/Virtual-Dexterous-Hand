from src.common import ROOT
from archive.低证据实验审计.实验03_手套标定映射对比.compare_glove_calibration import compare_calibrations


def test_personalized_mapping_experiment_uses_manifest_validation_split():
    data = ROOT / "数据手套数据"
    result = compare_calibrations(
        data / "captured_samples.xlsx",
        data / "base_mapping.json",
        data / "manifest.json",
    )
    assert result["evaluation_protocol"] == "per_pose_fourth_frame"
    assert result["validation_samples"] == 6
    assert result["output_dimensions"] == 6
    assert result["personalized_mapping"]["rmse_deg"] < result["base_mapping"]["rmse_deg"]
