import os
import subprocess

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
    clean_path = file_path.lstrip("/")
    if clean_path.startswith(f"{repo_name}/"):
        clean_path = clean_path[len(repo_name) + 1 :]

    clean_path = clean_path.replace("app/repos/", "").replace("repos/", "")
    if clean_path.startswith(f"{repo_name}/"):
        clean_path = clean_path[len(repo_name) + 1 :]

    base_path = os.path.join(os.getcwd(), "repos", repo_name)
    full_path = os.path.normpath(os.path.join(base_path, clean_path))

    if not full_path.startswith(os.path.abspath("repos")):
        return f"ERROR: Access Denied. {full_path} is outside allowed directory."

    try:
        if not os.path.exists(full_path):
            return f"ERROR: File not found at {clean_path}. Please check 'list_directory_structure' for the exact relative path."

        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(15000)
    except Exception as e:
        return f"ERROR: {str(e)}"


@tool
def install_and_develop_tool(setup_command: str, tool_name: str, execution_command: str):
    """
    Evolves the agent by installing ANY system package or running setup scripts,
    then defining a command to use the new capability.

    Args:
        setup_command: The bash command to install the tool (e.g., 'apt-get update && apt-get install -y cppcheck')
        tool_name: A simple name for this new capability.
        execution_command: The command to run the tool, using '{path}' as the target placeholder.
                           Example: 'cppcheck {path}' or 'grep -r "TODO" {path}'
    """
    try:
        # 1. Execute setup (apt-get, pip, curl, etc.)
        # Using shell=True allows pipes and complex install strings
        install_proc = subprocess.run(setup_command, shell=True, capture_output=True, text=True)

        if install_proc.returncode != 0:
            return f"Evolution Error during setup: {install_proc.stderr}"

        # 2. Define the dynamic execution logic
        def dynamic_tool(target_path: str):
            # Resolve placeholders
            cmd_string = execution_command.replace("{path}", target_path)

            # Execute the newly installed tool
            result = subprocess.run(cmd_string, shell=True, capture_output=True, text=True)

            output = result.stdout if result.returncode == 0 else result.stderr
            return output if output else "Tool executed successfully but returned no output."

        dynamic_tool.__doc__ = f"Specialized tool '{tool_name}' installed. Command: {execution_command}"
        DEVELOPED_TOOLS[tool_name] = dynamic_tool

        return f"Successfully added tool: '{tool_name}' to the available tool registry."

    except Exception as e:
        return f"Tool installation failed: {str(e)}"
