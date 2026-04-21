from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from stem_cell_coding_agent.agent.state import StemState
from stem_cell_coding_agent.agent.tools import (
    install_and_develop_tool,
    list_directory_structure,
)


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


def evolution_phase(state: StemState):
    """The agent decides which tools to 'grow' or acknowledges their existence."""
    llm = ChatOpenAI(model="gpt-4o")

    # 1. Check if we've already tried to install something
    has_installed_tools = any(
        "Successfully evolved" in m.content for m in state["messages"] if hasattr(m, "content")
    )

    if has_installed_tools:
        # If we have successful installations in history, change the prompt
        # to force the agent to finish rather than repeat.
        prompt = f"""
        You have successfully installed the tools for your specialization ({state['specialization']}).
        Look at your message history to see what was installed.
        
        DO NOT call 'install_and_develop_tool' again for the same tools.
        
        Confirm that you are ready by listing your new tools and say exactly 'EVOLUTION COMPLETE'.
        """
    else:
        # Initial installation prompt
        prompt = f"""
        You are now a {state['specialization']}. 
        Based on your reasoning: {state['reasoning']}, what CLI tools do you need?
        Call the 'install_and_develop_tool' to get them.
        Common tools: 'bandit' (security), 'black' (formatting), 'pylint' (quality).
        Only install 1 or 2 essential tools.
        """

    # We still bind the tools because the LLM might decide it needs ONE more
    # tool before it's done, but the conditional prompt discourages repeats.
    response = llm.bind_tools([install_and_develop_tool]).invoke(
        [SystemMessage(content=prompt)] + state["messages"]
    )
    return {"messages": [response]}
