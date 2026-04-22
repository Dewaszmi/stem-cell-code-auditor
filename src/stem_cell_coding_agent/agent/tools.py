import os
import subprocess
import sys

from langchain_core.tools import tool

from . import DEVELOPED_TOOLS


@tool
def list_directory_structure(repo_path: str):
    """Recursively lists the file structure of the repository."""
    structure = []
    base_path = os.path.join("repos", repo_path)

    for root, dirs, files in os.walk(base_path):
        if any(x in root for x in [".git", "node_modules", "__pycache__", "dist", "build"]):
            continue
        level = root.replace(base_path, "").count(os.sep)
        indent = " " * 4 * level
        structure.append(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in files:
            structure.append(f"{subindent}{f}")

    return "\n".join(structure)


@tool
def read_file_content(repo_name: str, file_path: str):
    """Reads a file. file_path should be relative to the repo root."""
    # 1. Path Cleanup: If the agent includes the repo name in the path, strip it.
    clean_path = file_path.replace(f"{repo_name}/", "", 1)

    base_path = os.path.join("repos", repo_name)
    full_path = os.path.abspath(os.path.join(base_path, clean_path))

    # 2. Safety: Ensure it stays inside /repos
    if not full_path.startswith(os.path.abspath("repos")):
        return "ERROR: Access denied. Stay inside the repository."

    # 3. Graceful Failure: Try/Except to prevent graph crashes
    try:
        if not os.path.exists(full_path):
            return f"ERROR: File '{file_path}' not found. Check the directory structure and try again."

        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            # 4. Context Protection: Only read the first 10k characters
            content = f.read(10000)
            if len(content) == 10000:
                content += "\n... [File Truncated for Brevity] ..."
            return content
    except Exception as e:
        return f"ERROR: Could not read file: {str(e)}"


@tool
def install_and_develop_tool(package_name: str, tool_name: str, command_template: str):
    """
    Installs a python package and creates a new tool for the agent.
    IMPORTANT: The command_template must use '{path}' as the placeholder for the file or directory.
    Example: command_template='bandit -r {path}'
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

        def dynamic_tool(target_path: str):
            # Using .replace is "Defensive Programming"
            # It ignores {url} or other keys that would cause a KeyError
            cmd_string = command_template.replace("{path}", target_path)

            # Security: Basic check to ensure the agent isn't trying to escape the shell
            result = subprocess.run(cmd_string.split(), capture_output=True, text=True)
            return result.stdout if result.returncode == 0 else result.stderr

        dynamic_tool.__doc__ = f"Executes {tool_name} using {package_name} on the specified path."
        DEVELOPED_TOOLS[tool_name] = dynamic_tool

        return f"Successfully evolved! I now have the tool: {tool_name}. Note: I can only execute this on local paths."
    except Exception as e:
        return f"Evolution failed: {str(e)}"
