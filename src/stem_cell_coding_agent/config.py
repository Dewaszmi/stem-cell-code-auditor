import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPOS_DIR = f"{PROJECT_ROOT}/repos"
os.makedirs(REPOS_DIR, exist_ok=True)
