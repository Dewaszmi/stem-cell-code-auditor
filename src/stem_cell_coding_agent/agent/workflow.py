import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from stem_cell_coding_agent.agent.phases import evolution_phase, sensing_phase
from stem_cell_coding_agent.agent.tools import install_and_develop_tool


class StemState(TypedDict):
    repo_name: str
    messages: Annotated[List[BaseMessage], operator.add]
    specialization: str
    reasoning: str


def run_stem_agent(repo_path):
    workflow = StateGraph(StemState)

    workflow.add_node("sense_environment", sensing_phase)
    workflow.add_node("evolution_phase", evolution_phase)

    tool_node = ToolNode([install_and_develop_tool])
    workflow.add_node("execute_evolution_tools", tool_node)

    workflow.set_entry_point("sense_environment")
    workflow.add_edge("sense_environment", "evolution_phase")

    workflow.add_conditional_edges(
        "evolution_phase",
        tools_condition,
        {
            "tools": "execute_evolution_tools",  # If AI wants to install something
            END: END,  # If AI is done and just talking
        },
    )

    workflow.add_edge("execute_evolution_tools", "evolution_phase")

    app = workflow.compile()

    print(f"Stem Agent initializing in: {repo_path}...")

    initial_state = {
        "repo_name": repo_path,
        "messages": [],
        "specialization": "None",
        "reasoning": "None",
    }

    final_output = app.invoke(initial_state)

    print("\n--- DIFFERENTIATION COMPLETE ---")
    print(f"IDENTIFIED AS: {final_output['specialization']}")
    print(f"RATIONALE: {final_output['reasoning']}")
    print(f"FINAL REPORT: {final_output['messages'][-1].content}")

    return final_output
