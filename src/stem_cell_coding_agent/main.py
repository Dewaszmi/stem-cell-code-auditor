from dotenv import load_dotenv

from stem_cell_coding_agent.git_tools import clone_repo
from stem_cell_coding_agent.run_agent import run_stem_agent


def main():
    load_dotenv()

    repo_path = clone_repo()
    print(f"Stem Agent initializing in: {repo_path}...")
    run_stem_agent(repo_path)


if __name__ == "__main__":
    main()
