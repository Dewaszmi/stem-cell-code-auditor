import os

from git import Repo

from stem_cell_coding_agent.config import REPOS_DIR


def clone_repo(repo_url="https://github.com/Dewaszmi/carjacker"):
    repo_name = repo_url.rsplit("/", 1)[1]
    repo_path = f"{REPOS_DIR}/{repo_name}"
    if not os.path.exists(repo_path):
        Repo.clone_from(repo_url, repo_path)
        print(f"Cloned repository: {repo_url} to {repo_path}")

    return repo_path
