import os
import subprocess
import sys

from langchain_core.tools import tool

from . import DEVELOPED_TOOLS


@tool
def list_directory_structure(repo_path: str):
    """
    Recursively lists the file structure of the repository.

    Used in the first phase of the agent's cycle, when transforming from a general code inspector to a specialized subclass.
    """
    structure = []
    # Using a subset of the path for safety/simplicity
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
def read_file_content(file_path: str):
    """Reads the actual code within a specific file."""
    with open(file_path, "r") as f:
        return f.read()


@tool
def install_and_develop_tool(package_name: str, tool_name: str, command_template: str):
    """
    Installs a python package and creates a new tool for the agent.
    Example: package_name='bandit', tool_name='security_scan', command_template='bandit -r {path}'
    """
    try:
        # 1. Install the package inside the Docker container
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

        # 2. Define the new tool logic on the fly
        def dynamic_tool(target_path: str):
            cmd = command_template.format(path=target_path)
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            return result.stdout if result.returncode == 0 else result.stderr

        # 3. Add to our internal registry
        dynamic_tool.__doc__ = f"Executes {tool_name} using {package_name} on the specified path."
        DEVELOPED_TOOLS[tool_name] = dynamic_tool

        return f"Successfully evolved! I now have the tool: {tool_name}"
    except Exception as e:
        return f"Evolution failed: {str(e)}"
