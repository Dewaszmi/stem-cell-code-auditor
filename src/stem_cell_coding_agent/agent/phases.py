import re

from langchain_core.messages import SystemMessage, trim_messages
from langchain_openai import ChatOpenAI

from stem_cell_coding_agent.agent.state import StemState
from stem_cell_coding_agent.agent.tools import (
    install_and_develop_tool,
    list_directory_structure,
    read_file_content,
)

from . import DEVELOPED_TOOLS


def sensing_phase(state: StemState):
    """The agent explores the codebase to decide its fate."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = f"""
    You are a General Code Auditor that wants to evolve. 
    Target Repository: {state['repo_name']}
    
    1. Explore the codebase using tools.
    2. Once you know what you want to become, you MUST respond exactly in this format:
       SPECIALIZATION: [Your Choice]
       REASONING: [Your Rationale]
    """

    tools = [list_directory_structure, read_file_content]
    # We use trimmed messages to avoid Context Overflow
    response = llm.bind_tools(tools).invoke([SystemMessage(content=prompt)] + state["messages"])

    # --- CRITICAL: Only parse if NOT calling a tool ---
    # If the agent is calling a tool, we return the message so the graph can route to tools.
    if not response.tool_calls:
        try:
            content = response.content
            spec = re.search(r"SPECIALIZATION:\s*(.*)", content, re.I).group(1).split("\n")[0].strip()
            reason = re.search(r"REASONING:\s*(.*)", content, re.I | re.S).group(1).strip()
            return {"specialization": spec, "reasoning": reason, "messages": [response]}
        except:
            pass  # Let it fall through to just returning the message

    return {"messages": [response]}


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
        If deemed necessary, you can install another tool. If you think you are ready, confirm by listing your new tools and say exactly "TOOL INSTALLATION COMPLETE"
        """
    else:
        # Initial installation prompt
        prompt = f"""
        You are now a {state['specialization']}. 
        Based on your reasoning: {state['reasoning']}, what code audition tools do you need?
        Call the 'install_and_develop_tool' to get them.
        """

    # We still bind the tools because the LLM might decide it needs ONE more
    # tool before it's done, but the conditional prompt discourages repeats.
    response = llm.bind_tools([install_and_develop_tool]).invoke(
        [SystemMessage(content=prompt)] + state["messages"]
    )
    return {"messages": [response]}


def generalist_audit_phase(state: StemState):
    """The untrained agent exploring and auditing."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = f"""
    You are a General Code Auditor. 
    Target Repository: {state['repo_name']}
    
    Your task:
    Utilise the tools at your disposal ('list_directory_structure', 'read_file_content') to audit the code.
    Detect issues and problems (purposeful or not).
    
    If you are done auditing and gathered enough information, summarize your findings and respond with a final list of issues formatted as:
    ISSUES DETECTED:
    - [Issue 1]
    - [Issue 2]
    ...
    TOTAL ISSUE COUNT: [X]
    """

    # Bind only the basic vision tools
    tools = [list_directory_structure, read_file_content]
    response = llm.bind_tools(tools).invoke([SystemMessage(content=prompt)] + state["messages"])

    return {"messages": [response]}


def specialized_audit_phase(state: StemState):
    """The evolved agent auditing with custom tools."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = f"""
    You are a {state['specialization']} Code Auditor.
    Target Repository: {state['repo_name']}
    
    Your task:
    Utilise the tools at your disposal ('list_directory_structure', 'read_file_content', as well as any other tools installed in the tool installation phase) to audit the code.
    Detect issues and problems (purposeful or not).
    
    If you are done auditing and gathered enough information, summarize your findings and respond with a final list of issues formatted as:
    ISSUES DETECTED:
    - [Issue 1]
    - [Issue 2]
    ...
    TOTAL ISSUE COUNT: [X]
    """

    # Bind basic tools PLUS the dynamically developed tools
    dynamic_tools = list(DEVELOPED_TOOLS.values())
    all_tools = [list_directory_structure, read_file_content] + dynamic_tools

    response = llm.bind_tools(all_tools).invoke([SystemMessage(content=prompt)] + state["messages"])

    return {"messages": [response]}
