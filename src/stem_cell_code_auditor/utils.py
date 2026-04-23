import os
from pathlib import Path

from git import Repo

from stem_cell_code_auditor.config import REPOS_DIR


def clone_repo(repo_url):
    """Clones the repository if not exists, returns path to the repository."""
    repo_name = repo_url.rsplit("/", 1)[1]
    repo_path = f"{REPOS_DIR}/{repo_name}"
    if not os.path.exists(repo_path):
        Repo.clone_from(repo_url, repo_path)
        print(f"Cloned repository: {repo_url} to {repo_path}")

    return repo_path


# Source - https://stackoverflow.com/a/73564246
# Posted by miigotu, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-21, License - CC BY-SA 4.0


# Helper to check whether code is ran inside a Docker container
def is_docker():
    cgroup = Path("/proc/self/cgroup")
    return Path("/.dockerenv").is_file() or (cgroup.is_file() and "docker" in cgroup.read_text())
