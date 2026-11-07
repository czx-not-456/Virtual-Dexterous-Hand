"""兼容入口；研究实现位于 experiments/实验01_StageB七方法消融/evaluate_tqhg_v2_faults.py。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == '__main__':
    import runpy
    runpy.run_module('experiments.实验01_StageB七方法消融.evaluate_tqhg_v2_faults', run_name='__main__')
else:
    from experiments.实验01_StageB七方法消融.evaluate_tqhg_v2_faults import *
