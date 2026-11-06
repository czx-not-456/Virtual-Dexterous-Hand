"""兼容入口：实际实验代码已归档到 实验06_PCA与可达性协同对比/benchmark_reachability_synergy.py。
从项目根目录运行此文件的旧命令仍然可用。
"""
from pathlib import Path
import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if __name__ == '__main__':
    import runpy
    runpy.run_module('实验06_PCA与可达性协同对比.benchmark_reachability_synergy', run_name='__main__')
else:
    from 实验06_PCA与可达性协同对比.benchmark_reachability_synergy import *
