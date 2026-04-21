from dotenv import load_dotenv

from stem_cell_coding_agent.agent.workflow import run_stem_agent
from stem_cell_coding_agent.git_tools import clone_repo


def main():
    load_dotenv()

    repo_path = clone_repo()
    run_stem_agent(repo_path)


if __name__ == "__main__":
    main()
