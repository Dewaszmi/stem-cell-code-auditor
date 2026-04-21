import os
from typing import List, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# --- 1. THE VISION TOOLS (The Agent's "Eyes") ---


@tool
def list_directory_structure(repo_path: str):
    """Recursively lists the file structure of the repository."""
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


# --- 2. THE AGENT STATE (The Agent's "DNA") ---


class StemState(TypedDict):
    repo_name: str
    messages: List[BaseMessage]
    specialization: str
    reasoning: str


# --- 3. THE NODES (The Biological Phases) ---


def sensing_phase(state: StemState):
    """The Stem Cell looks at its environment."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Get the structure
    structure = list_directory_structure.invoke(state["repo_name"])

    prompt = f"""
    You are a Stem Cell AI. You have just been placed into a new codebase environment.
    Your goal is to sense the signals (files/folders) and decide what specialized 
    Auditor Agent you should become.

    DIRECTORY STRUCTURE:
    {structure}

    Analyze the files. Are there many .js/.tsx files? Are there SQL/Python files? 
    Is there a Dockerfile or a package.json?

    Respond in the following format:
    SPECIALIZATION: [Name of the specialist you will become]
    REASONING: [Brief explanation of why based on the files you saw]
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    # Parse the response (simple split for this demo)
    content = response.content
    spec = content.split("SPECIALIZATION:")[1].split("REASONING:")[0].strip()
    reason = content.split("REASONING:")[1].strip()

    return {"specialization": spec, "reasoning": reason, "messages": [response]}


def run_stem_agent(repo_path):

    # --- 4. THE GRAPH (The Life Cycle) ---

    workflow = StateGraph(StemState)

    # Add the node
    workflow.add_node("sense_environment", sensing_phase)

    # Start the process
    workflow.set_entry_point("sense_environment")

    # For now, we stop after sensing (differentiation complete)
    workflow.add_edge("sense_environment", END)

    app = workflow.compile()

    # --- 5. EXECUTION ---
    initial_state = {
        "repo_name": repo_path,
        "messages": [],
        "specialization": "None",
        "reasoning": "None",
    }

    final_state = app.invoke(initial_state)

    print("\n--- DIFFERENTIATION COMPLETE ---")
    print(f"IDENTIFIED AS: {final_state['specialization']}")
    print(f"RATIONALE: {final_state['reasoning']}")
