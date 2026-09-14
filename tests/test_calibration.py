from src.glove.calibration import AdaptiveCalibrator
from src.glove.mock_driver import MockGloveDriver


def test_mock_calibration_maps_into_unit_interval():
    channels = ["thumb_mcp", "thumb_pip", "thumb_dip", "index_mcp", "index_pip", "index_dip",
                "middle_mcp", "middle_pip", "middle_dip", "ring_mcp", "ring_pip", "ring_dip",
                "little_mcp", "little_pip", "little_dip"]
    d = MockGloveDriver(channels)
    c = AdaptiveCalibrator(channels)
    c.fit(d.calibration_samples())
    n = c.normalize(d.sample())
    assert all(0.0 <= x <= 1.0 for x in n.values())
