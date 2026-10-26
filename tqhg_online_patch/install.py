"""Safe, reversible installer for online TQHG on the servo-ablation edition."""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from patch_main import patch

BACKUP = ROOT / "_tqhg_online_backup"
NEW_FILES = [
    "src/retargeting/tqhg_online.py",
    "src/retargeting/task_weighted_hybrid.py",
    "scripts/train_tqhg_online.py",
    "test/tests_20261012/test_tqhg_online.py",
]


def install():
    main_path = ROOT / "src/main.py"
    if not main_path.exists():
        raise SystemExit("[ERROR] install.py should be inside project_root/tqhg_online_patch/")
    if BACKUP.exists():
        raise SystemExit("[ERROR] Backup directory already exists: don't overwrite; use --restore first")
    try:
        updated = patch(main_path.read_text(encoding="utf-8"))
        compile(updated, str(main_path), "exec")
        for rel in NEW_FILES:
            p = HERE / 'payload' / rel
            if not p.is_file():
                raise FileNotFoundError(p)
        BACKUP.mkdir(parents=True)
        shutil.copy2(main_path, BACKUP / "main.py")
        for rel in NEW_FILES:
            dest = ROOT / rel
            marker = BACKUP / (rel.replace('/', '__') + '.existing')
            if dest.exists():
                shutil.copy2(dest, marker)
        # Use atomic swap only after all checks and backups have passed.
        temp = main_path.with_suffix('.tqhg_pending')
        temp.write_text(updated, encoding="utf-8")
        temp.replace(main_path)
        for rel in NEW_FILES:
            dest = ROOT / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(HERE / 'payload' / rel, dest)
        print("[OK] TQHG online integration installed; original src/main.py backed up")
        print("[NEXT] python scripts/train_tqhg_online.py")
    except Exception:
        if BACKUP.exists():
            restore()
        raise


def restore():
    if not BACKUP.exists():
        raise SystemExit("[ERROR] No TQHG backup found")
    main_path = ROOT / "src/main.py"
    shutil.copy2(BACKUP / 'main.py', main_path)
    for rel in NEW_FILES:
        dest = ROOT / rel
        marker = BACKUP / (rel.replace('/', '__') + '.existing')
        if marker.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(marker, dest)
        elif dest.exists():
            dest.unlink()
    shutil.rmtree(BACKUP)
    print("[OK] Restored previous code; calibration JSON kept for reproducibility")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--restore', action='store_true')
    args = p.parse_args()
    restore() if args.restore else install()
