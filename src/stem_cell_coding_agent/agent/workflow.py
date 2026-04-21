from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from stem_cell_coding_agent.agent.phases import (
    evolution_phase,
    generalist_audit_phase,
    sensing_phase,
    specialized_audit_phase,
)
from stem_cell_coding_agent.agent.state import StemState
from stem_cell_coding_agent.agent.tools import (
    install_and_develop_tool,
    list_directory_structure,
    read_file_content,
)

from . import DEVELOPED_TOOLS


def run_generalist_agent(repo_path: str):
    workflow = StateGraph(StemState)

    # Nodes
    workflow.add_node("audit", generalist_audit_phase)
    # The ToolNode acts as the hands for the generalist
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

    # Nodes
    workflow.add_node("sense_environment", sensing_phase)
    workflow.add_node("evolution_phase", evolution_phase)
    workflow.add_node("execute_evolution_tools", ToolNode([install_and_develop_tool]))
    workflow.add_node("audit_phase", specialized_audit_phase)

    def dynamic_tool_node(state: StemState):
        current_tools = [list_directory_structure, read_file_content] + list(DEVELOPED_TOOLS.values())
        node = ToolNode(current_tools)
        return node.invoke(state)

    workflow.add_node("audit_tools", dynamic_tool_node)

    workflow.set_entry_point("sense_environment")

    # Edges
    workflow.add_edge("sense_environment", "evolution_phase")

    # Evolution loop
    workflow.add_conditional_edges(
        "evolution_phase",
        tools_condition,
        {"tools": "execute_evolution_tools", END: "audit_phase"},
    )

    workflow.add_edge("execute_evolution_tools", "evolution_phase")

    # Audit loop
    workflow.add_conditional_edges(
        "audit_phase", tools_condition, {"tools": "audit_tools", END: END}  # When done auditing, finish
    )
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

    # --- UPDATED PRINT STATEMENTS ---
    print("\n--- STEM AGENT AUDIT COMPLETE ---")
    print(f"IDENTIFIED AS: {final_state['specialization']}")
    print(f"RATIONALE: {final_state['reasoning']}")

    # Adding a newline before the report makes it easier to read in the terminal
    print(f"\nFINAL REPORT:\n{final_state['messages'][-1].content}")
    return final_state
