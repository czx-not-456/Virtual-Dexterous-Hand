"""兼容入口：实际实验代码已归档到 实验09_StageC五方法MuJoCo对比/calibrate_stageC_task_gate.py。
从项目根目录运行此文件的旧命令仍然可用。
"""
from pathlib import Path
import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if __name__ == '__main__':
    import runpy
    runpy.run_module('experiments.实验09_StageC五方法MuJoCo对比.calibrate_stageC_task_gate', run_name='__main__')
else:
    from experiments.实验09_StageC五方法MuJoCo对比.calibrate_stageC_task_gate import *
