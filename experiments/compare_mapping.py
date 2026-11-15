"""历史兼容入口：已移至 archive.低证据实验审计.实验02_静态与动态映射对比.compare_mapping（非核心论文实验）。"""
from pathlib import Path
import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if __name__ == '__main__':
    import runpy
    runpy.run_module('archive.低证据实验审计.实验02_静态与动态映射对比.compare_mapping', run_name='__main__')
else:
    from archive.低证据实验审计.实验02_静态与动态映射对比.compare_mapping import *
