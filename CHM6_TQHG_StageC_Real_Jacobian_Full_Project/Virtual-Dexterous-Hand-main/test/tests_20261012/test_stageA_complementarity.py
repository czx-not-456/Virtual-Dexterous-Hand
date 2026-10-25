"""Fast pure unit tests for stage-A pairing/statistics; no Mujoco."""
from scripts.analyze_expert_complementarity import classify, summarize


def test_oracle_never_worse_than_experts():
    records = []
    for i, (a, b, mix) in enumerate(((0.1, 0.2, 0.15), (0.3, 0.1, 0.12), (0.2, 0.2, 0.2))):
        records.append({"pose": "P", "sample_index": i,
                        "ridge6_mae": a, "robust4_mae": b, "tqhg_v2_mae": mix,
                        "oracle_mae": min(a, b), "winner": classify(a, b),
                        "w_task": 0., "w_null": .4, "sensor_reliability": .8})
    r = summarize(records)
    assert r["oracle_mae"] <= min(r["ridge6_mae"], r["robust4_mae"])
    assert r["ridge6_win_pct"] == 100 / 3
    assert r["robust4_win_pct"] == 100 / 3
    assert r["tie_pct"] == 100 / 3
    assert r["oracle_gain_over_best"] > 0
    assert r["tqhg_regret_vs_oracle"] >= 0


def test_ties_have_numeric_tolerance():
    assert classify(.1, .10000001, 1e-6) == "tie"
    assert classify(.1, .1001, 1e-6) == "ridge6"
    assert classify(.1001, .1, 1e-6) == "robust4"
