"""Compatibility entry point for the focused Scheme E research ablation."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.实验08_任务效应谱方案对比.benchmark_scheme_e_research import main
if __name__ == "__main__":
    raise SystemExit(main())
