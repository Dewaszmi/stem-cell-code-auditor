import sys

from dotenv import load_dotenv

from stem_cell_coding_agent.agent.workflow import run_generalist_agent, run_stem_agent
from stem_cell_coding_agent.git_tools import clone_repo
from stem_cell_coding_agent.utils import is_docker


def main():
    if not is_docker():
        print(
            "It's possible that program isn't being ran in a Docker container, the program will now exit for the sake of system safety."
        )
        sys.exit(1)

    if len(sys.argv) < 2:
        print("No repository has been provided via the CLI arguments, exiting.")
        sys.exit(1)

    repo_path = sys.argv[1]

    load_dotenv()

    repo_path = clone_repo(repo_url=repo_path)
    gen_final = run_generalist_agent(repo_path)
    stem_final = run_stem_agent(repo_path)

    import re

    def get_count(message):
        match = re.search(r"TOTAL ISSUE COUNT:\s*(\d+)", message)
        return match.group(1) if match else "Unknown"

    # After running both:
    print(f"Generalist found: {get_count(gen_final['messages'][-1].content)}")
    print(f"Stem Agent found: {get_count(stem_final['messages'][-1].content)}")


if __name__ == "__main__":
    main()
