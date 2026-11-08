"""兼容入口；研究实现位于 experiments/StageA_专家互补性/analyze_expert_complementarity.py。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == '__main__':
    import runpy
    runpy.run_module('experiments.StageA_专家互补性.analyze_expert_complementarity', run_name='__main__')
else:
    from experiments.StageA_专家互补性.analyze_expert_complementarity import *
