from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from . import DEVELOPED_TOOLS
from .phases import (
    evolution_phase,
    generalist_audit_phase,
    sensing_phase,
    specialized_audit_phase,
)
from .state import StemState
from .tools import install_and_develop_tool, list_directory_structure, read_file_content


def run_generalist_agent(repo_path: str):
    workflow = StateGraph(StemState)

    # Nodes
    workflow.add_node("audit", generalist_audit_phase)
    workflow.add_node("tools", ToolNode([list_directory_structure, read_file_content]))

    # Edges
    workflow.set_entry_point("audit")

    # If the LLM calls a tool, go to tools. Otherwise, END.
    workflow.add_conditional_edges("audit", tools_condition, {"tools": "tools", END: END})
    workflow.add_edge("tools", "audit")

    app = workflow.compile()

    print(f"🤖 Generalist Agent starting audit on: {repo_path}...")
    final_state = app.invoke(
        {"repo_name": repo_path, "messages": [], "specialization": "Generalist", "reasoning": "None"}
    )

    print("\n--- GENERALIST AUDIT COMPLETE ---")
    print(final_state["messages"][-1].content)
    return final_state


def run_stem_agent(repo_path):
    workflow = StateGraph(StemState)

    # 1. Add Nodes
    workflow.add_node("sensing_node", sensing_phase)  # Renamed for clarity
    workflow.add_node("base_tools", ToolNode([list_directory_structure, read_file_content]))
    workflow.add_node("evolution_phase", evolution_phase)
    workflow.add_node("execute_evolution_tools", ToolNode([install_and_develop_tool]))
    workflow.add_node("audit_phase", specialized_audit_phase)

    def audit_tool_node(state: StemState):
        current_tools = [list_directory_structure, read_file_content] + list(DEVELOPED_TOOLS.values())
        return ToolNode(current_tools).invoke(state)

    workflow.add_node("audit_tools", audit_tool_node)

    workflow.set_entry_point("sensing_node")

    # Sensing Loop: Loop to tools, or proceed to evolution when text is returned
    workflow.add_conditional_edges(
        "sensing_node",
        tools_condition,
        {"tools": "base_tools", END: "evolution_phase"},
    )
    workflow.add_edge("base_tools", "sensing_node")

    # Evolution Loop
    workflow.add_conditional_edges(
        "evolution_phase",
        tools_condition,
        {"tools": "execute_evolution_tools", END: "audit_phase"},
    )
    workflow.add_edge("execute_evolution_tools", "evolution_phase")

    # Audit Loop
    workflow.add_conditional_edges("audit_phase", tools_condition, {"tools": "audit_tools", END: END})
    workflow.add_edge("audit_tools", "audit_phase")

    app = workflow.compile()

    # Execution
    print(f"Stem Agent initializing in: {repo_path}...")

    initial_state = {
        "repo_name": repo_path,
        "messages": [],
        "specialization": "None",
        "reasoning": "None",
    }

    final_state = app.invoke(initial_state)

    print("\n--- STEM AGENT AUDIT COMPLETE ---")
    print(f"IDENTIFIED AS: {final_state['specialization']}")
    print(f"RATIONALE: {final_state['reasoning']}")

    print(f"\nFINAL REPORT:\n{final_state['messages'][-1].content}")
    return final_state
