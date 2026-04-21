from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from stem_cell_coding_agent.agent.state import StemState
from stem_cell_coding_agent.agent.tools import (
    install_and_develop_tool,
    list_directory_structure,
    read_file_content,
)

from . import DEVELOPED_TOOLS


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


def generalist_audit_phase(state: StemState):
    """The untrained agent exploring and auditing."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = f"""
    You are a General Code Auditor. 
    Target Repository: {state['repo_name']}
    
    Your task:
    1. Use 'list_directory_structure' to see what's in the repo.
    2. Use 'read_file_content' to read files that look suspicious or important.
    3. If you have gathered enough information, summarize your findings.
    
    If you are done exploring, respond with a final list of issues formatted as:
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
    You are a fully evolved {state['specialization']}.
    Target Repository: {state['repo_name']}
    
    Your task:
    1. You have your basic sight ('list_directory_structure', 'read_file_content').
    2. MORE IMPORTANTLY: You must use the tools you installed during your evolution phase to audit the code.
    3. Combine your manual reading with your specialized tool outputs.
    
    If you are done auditing, respond with a final list of issues formatted as:
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
