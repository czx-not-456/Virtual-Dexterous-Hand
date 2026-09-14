from src.common import load_yaml
from src.features.feature_extractor import FeatureExtractor
from src.features.intent_recognition import Intent, IntentRecognizer
from src.glove.calibration import AdaptiveCalibrator
from src.glove.driver import GloveSample
from src.glove.mock_driver import MockGloveDriver
from src.hand_model.human_hand import HumanHandModel


def _features_for(gesture: str):
    gc = load_yaml("configs/glove.yaml")
    d = MockGloveDriver(gc["channels"])
    c = AdaptiveCalibrator(gc["channels"])
    c.fit(d.calibration_samples())
    n = d._gesture_vector(gesture)
    s = GloveSample(0.0, d.raw_from_normalized(n), (0.0, 0.0, 0.0))
    h = HumanHandModel()
    return FeatureExtractor(h).extract(h.angles_from_normalized(c.normalize(s)))


def test_wrap_score_is_high_for_wrap():
    assert _features_for("WRAP").wrap_score > _features_for("OPEN").wrap_score


def test_intent_hysteresis_reaches_wrap():
    ac = load_yaml("configs/algorithm.yaml")["intent"]
    r = IntentRecognizer(ac["pinch_enter_m"], ac["pinch_exit_m"], ac["wrap_enter"], ac["wrap_exit"], 2, 1)
    f = _features_for("WRAP")
    r.update(f)
    state = r.update(f)
    assert state == Intent.WRAP


def test_temporal_window_rejects_single_wrap_spike():
    ac = load_yaml("configs/algorithm.yaml")["intent"]
    r = IntentRecognizer(ac["pinch_enter_m"], ac["pinch_exit_m"], ac["wrap_enter"], ac["wrap_exit"], 2, 5)
    neutral = _features_for("OPEN")
    wrap = _features_for("WRAP")
    for _ in range(5):
        assert r.update(neutral) == Intent.NEUTRAL
    # 单帧异常不应立即进入 WRAP
    assert r.update(wrap) == Intent.NEUTRAL
