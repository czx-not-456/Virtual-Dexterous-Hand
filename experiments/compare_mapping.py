from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_yaml
from src.features.feature_extractor import FeatureExtractor
from src.features.intent_recognition import Intent
from src.glove.calibration import AdaptiveCalibrator
from src.glove.driver import GloveSample
from src.glove.mock_driver import MockGloveDriver
from src.hand_model.human_hand import HumanHandModel
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.constraints import coupling_error, fingertip_collision_penalty, vector_shape_error
from src.retargeting.optimizer import IntentDrivenRetargeter


def pct_improvement(baseline: float, improved: float) -> float:
    return (baseline - improved) / max(abs(baseline), 1e-12) * 100.0


def main() -> None:
    gc = load_yaml("configs/glove.yaml")
    ac = load_yaml("configs/algorithm.yaml")
    driver = MockGloveDriver(gc["channels"], sampling_hz=gc["sampling_hz"])
    cal = AdaptiveCalibrator(gc["channels"])
    cal.fit(driver.calibration_samples())
    human = HumanHandModel()
    robot = RobotHandModel()
    extractor = FeatureExtractor(human)
    solver = IntentDrivenRetargeter(human, robot, ac["optimizer"])

    metrics = {}
    for gesture, intent in [("PINCH", Intent.PINCH), ("WRAP", Intent.WRAP)]:
        rows = {k: [] for k in ["s_pos", "d_pos", "s_vec", "d_vec", "s_cpl", "d_cpl", "s_col", "d_col"]}
        qps, qpd = None, None
        for _ in range(80):
            sample = GloveSample(0.0, driver.raw_from_normalized(driver._gesture_vector(gesture)), (0.0, 0.0, 0.0))
            qh = human.angles_from_normalized(cal.normalize(sample))
            f = extractor.extract(qh)
            rs = solver.solve(qh, f.fingertip_positions, intent, qps, dynamic=False)
            rd = solver.solve(qh, f.fingertip_positions, intent, qpd, dynamic=True)
            qps, qpd = rs.q_target, rd.q_target
            for prefix, r in [("s", rs), ("d", rd)]:
                tips = robot.fingertips(r.q_target)
                rows[f"{prefix}_pos"].append(solver._position_error(r.q_target, f.fingertip_positions, intent))
                rows[f"{prefix}_vec"].append(vector_shape_error(tips, f.fingertip_positions, solver.task_scale, solver.vector_pairs))
                rows[f"{prefix}_cpl"].append(coupling_error(r.q_target, robot))
                rows[f"{prefix}_col"].append(fingertip_collision_penalty(r.q_target, robot, intent.value))
        metrics[gesture] = {k: float(np.mean(v)) for k, v in rows.items()}

    p = metrics["PINCH"]
    w = metrics["WRAP"]
    print("Intent-aware CH-M6 15-to-11 mapping comparison")
    print("- PINCH: prioritizes thumb-index endpoint/vector alignment while preserving collision safety.")
    print("- WRAP: prioritizes CH-M6 multi-finger coordination and whole-hand shape.")
    print()
    print(
        f"PINCH position MSE: static={p['s_pos']:.8f}, dynamic={p['d_pos']:.8f}, "
        f"improvement={pct_improvement(p['s_pos'], p['d_pos']):.2f}%"
    )
    print(
        f"PINCH vector MSE:   static={p['s_vec']:.8f}, dynamic={p['d_vec']:.8f}, "
        f"improvement={pct_improvement(p['s_vec'], p['d_vec']):.2f}%"
    )
    print(
        f"WRAP configured synergy penalty: static={w['s_cpl']:.8f}, dynamic={w['d_cpl']:.8f}, "
        f"improvement={pct_improvement(w['s_cpl'], w['d_cpl']):.2f}%"
    )
    print(
        f"WRAP vector MSE:       static={w['s_vec']:.8f}, dynamic={w['d_vec']:.8f}, "
        f"improvement={pct_improvement(w['s_vec'], w['d_vec']):.2f}%"
    )
    print()
    print("These are mock-data engineering checks, not final experimental results for the project report/paper.")


if __name__ == "__main__":
    main()
