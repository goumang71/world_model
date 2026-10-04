"""Use the baseline dependency environment from the requested Python entry point."""
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    python = root / '.venv' / 'Scripts' / 'python.exe'
    raise SystemExit(subprocess.call([str(python), str(root / 'evaluate_baseline.py'), *sys.argv[1:]], cwd=root))
