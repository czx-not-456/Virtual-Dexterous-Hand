from src.common import load_yaml
from src.features.feature_extractor import FeatureExtractor
from src.glove.calibration import AdaptiveCalibrator
from src.glove.mock_driver import MockGloveDriver
from src.hand_model.human_hand import HumanHandModel


def test_mock_pipeline_smoke():
    gc = load_yaml("configs/glove.yaml")
    d = MockGloveDriver(gc["channels"])
    cal = AdaptiveCalibrator(gc["channels"])
    cal.fit(d.calibration_samples(5))
    h = HumanHandModel()
    ext = FeatureExtractor(h)
    for _ in range(10):
        norm = cal.normalize(d.sample())
        q = h.angles_from_normalized(norm)
        f = ext.extract(q)
        assert f.joint_angles.shape == (15,)
