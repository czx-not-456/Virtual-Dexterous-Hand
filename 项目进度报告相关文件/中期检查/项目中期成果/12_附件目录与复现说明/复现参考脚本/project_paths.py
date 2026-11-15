from pathlib import Path

HERE = Path(__file__).resolve().parent

def find_root(start=HERE):
    for p in (Path(start).resolve(), *Path(start).resolve().parents):
        if (p/"pyproject.toml").is_file() and (p/"src").is_dir() and (p/"datasets").is_dir():
            return p
    raise FileNotFoundError("未找到项目根目录（需 pyproject.toml、src 和 datasets）。请将成果目录放在完整项目内部。")

ROOT = find_root()
