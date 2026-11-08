"""Reproduce the 2026-10-08 review checks from the project root."""
from pathlib import Path
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
os.chdir(ROOT)
env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
OUT.mkdir(exist_ok=True)
def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8").stdout.strip()
tracked = subprocess.run(["git", "ls-files", "-z"], capture_output=True).stdout.decode("utf-8").split("\0")
metadata = {"started_at": datetime.now().astimezone().isoformat(), "python": sys.version,
            "executable": sys.executable, "platform": platform.platform(), "commit": git("rev-parse", "HEAD"),
            "initial_git_status": git("status", "--short"),
            "packages": {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "PyYAML", "mujoco", "pytest"]},
            "sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in tracked
                       if p and (ROOT / p).is_file() and (p.startswith(("src/", "configs/", "test/", "experiments/", "CH-M6/", "datasets/", "数据手套数据/")) or p in ["requirements.txt", "pyproject.toml"])}}
(OUT / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
checks = [
 ("01_pip_check", ["-m", "pip", "check"]),
 ("02_pytest", ["-m", "pytest", "-p", "no:cacheprovider", "-rA", "--junitxml=" + str(OUT / "pytest_results.xml"), "--basetemp=" + str(OUT / "pytest_tmp")]),
 ("03_mapping", ["experiments/compare_mapping.py"]),
 ("04_glove_calibration", ["experiments/compare_glove_calibration.py", "--output-dir", str(OUT / "glove_calibration")]),
 ("05_dataset_pipeline", ["-m", "src.main", "--sim", "none", "--input", "sim", "--sim-source", "dataset", "--steps", "800", "--task", "pinch", "--output-dir", str(OUT / "input_checks"), "--run-name", "dataset"]),
 ("06_mock_pipeline", ["-m", "src.main", "--sim", "none", "--input", "sim", "--sim-source", "mock", "--steps", "800", "--task", "wrap", "--output-dir", str(OUT / "input_checks"), "--run-name", "mock"]),
 ("07_glove_pipeline", ["-m", "src.main", "--sim", "none", "--input", "glove", "--steps", "24", "--task", "pinch", "--output-dir", str(OUT / "input_checks"), "--run-name", "glove"]),
 ("08_synthetic_benchmark", ["experiments/benchmark_synthetic.py", "--trials", "3", "--sim", "mujoco", "--output-dir", str(OUT / "benchmark")]),
 ("09_glove_tasks", ["experiments/compare_glove_tasks.py", "--steps", "240", "--output-dir", str(OUT / "glove_tasks")]),
]
results = []
for name, args in checks:
    command = [sys.executable, *args]
    started = time.perf_counter()
    print("START", name, flush=True)
    try:
        with (OUT / (name + ".log")).open("w", encoding="utf-8") as log:
            log.write("Command: " + subprocess.list2cmdline(command) + "\n\n")
            log.flush()
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        result = {"name": name, "command": command, "returncode": completed.returncode}
    except subprocess.TimeoutExpired:
        result = {"name": name, "command": command, "returncode": None, "error": "timeout after 1800 seconds"}
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    results.append(result)
    (OUT / "execution_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("END", name, result["returncode"], result["elapsed_seconds"], flush=True)
print("COMPLETE", flush=True)
