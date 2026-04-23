import re

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from . import DEVELOPED_TOOLS
from .state import StemState
from .tools import install_and_develop_tool, list_directory_structure, read_file_content


def sensing_phase(state: StemState):
    """The agent explores the codebase to decide its fate."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = f"""
    You are a General Code Auditor that's going to specialize in a more niche branch of code auditioning. 
    Target Repository: {state['repo_name']}
    
    1. Explore the codebase using given tools.
    2. Determine primary programming languages and frameworks.
    
    RESPOND IN THIS FORMAT:
       SPECIALIZATION: [Your Choice]
       TECH_STACK: [List languages and frameworks found, e.g., PHP, MySQL, Apache]
       REASONING: [Your Rationale, why this specialization fits this specific tech stack]
       
    For example:
    - Upon detecting substantial amount of backend code, APIs, SQL, sensitive logic - you might want to specialize into a "Security Hardener", focusing on OWASP Top 10, SQL Injection, Auth logic
    - Upon detecting substantial amount of frontend code, CSS/TSX, React/Vue - you might want to specialize into a "UX/Performance Optimizer", focusing on accessibility, bundle size, rendering bottlenecks.
    """

    print(f"\n{'='*20} SENSING PHASE: {state['repo_name']} {'='*20}")

    tools = [list_directory_structure, read_file_content]
    response = llm.bind_tools(tools).invoke([SystemMessage(content=prompt)] + state["messages"])

    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(
                f"🔎 [EXPLORING]: Agent is calling tool '{tool_call['name']}' with args: {tool_call['args']}"
            )
        return {"messages": [response]}
    else:
        try:
            print(f"🎯 [DECISION]: Agent has finished sensing. Parsing results...")
            content = response.content
            spec = re.search(r"SPECIALIZATION:\s*(.*)", content, re.I).group(1).split("\n")[0].strip()
            stack = re.search(r"TECH_STACK:\s*(.*)", content, re.I).group(1).split("\n")[0].strip()
            reason = re.search(r"REASONING:\s*(.*)", content, re.I | re.S).group(1).strip()
            print(f"✅ [PARSED SPECIALIZATION]: {spec}")
            print(f"🛠️ [PARSED TECH STACK]: {stack}")
            print(f"💡 [RATIONALE]: {reason[:100]}...")  # Print first 100 chars of reasoning

            return {"specialization": spec, "tech_stack": stack, "reasoning": reason, "messages": [response]}
        except Exception as e:
            print(f"❌ [PARSING ERROR]: Failed to parse decision. Raw content: {content[:200]}")
            pass

    return {"messages": [response]}


def evolution_phase(state: StemState):
    """The agent decides which tools to develop or acknowledges their existence."""
    llm = ChatOpenAI(model="gpt-4o")

    # Check if the agent already tried to install something
    has_installed_tools = any(
        "Successfully evolved" in m.content for m in state["messages"] if hasattr(m, "content")
    )

    if has_installed_tools:
        print("📝 Status: At least one tool installed. Agent is deciding if more are needed...")
        # Prompt after installing at least one tool
        prompt = f"""
        You have successfully installed tools. Now you must decide if you need MORE 
        or if you are ready. If ready, summarize newly acquired tools and say "TOOL INSTALLATION COMPLETE".
        """
    else:
        print("🌱 Status: No tools installed yet. Requesting initial toolset...")
        # Initial installation prompt
        prompt = f"""
        You are a {state['specialization']}, specializing in {state['tech_stack']}.
        
        CONTEXT:
        The repository uses: {state['tech_stack']}
        Your rationale: {state['reasoning']}
        
        You are in a Debian-based Linux environment. You have root access to install any tools necessary for your audit.
        
        If you need a tool, you can use apt-get, pip, curl etc.
        Define the execution command using '{{path}}' as the target.
        
        It is mandatory to install at least one professional-grade CLI tool (eg. bandit, semgrep, safety, system-level scanners, all depending on specialization type).
        It is also highly recommended to install more than one tools deemed useful. Be ambitious.
        
        RULES FOR INSTALLATION:
        1. PREFER API: Most tools (nikto, bandit, lynis, nmap) are available via 'apt-get install -y'
        2. PREFER PIP: Use 'pip install' for python-specific tools like 'semgrep'
        3. AVOID CPAN: Avoid using cpan, it is slow and unstable in Docker.
        4. COMBINE: Install all related tools in ONE call to avoid lock errors.
        Example: 'apt-get update && apt-get install -y nikto bandit lynis'
        Remember that the tools mentioned in the prompt are examples, and what you should install should be based solely on your specialization.
        5. Do not install 'helper' tools like curl or wget as standalone tools. If you need them, use them inside a single setup_command.
        If you have already tried a command and it failed, DO NOT repeat it.
        Try a different package manager or move on with the tools you have.
        6. Based on the context info, choose tools that match the language.
        - If PHP: use 'psalm' or 'progpilot'
        - If JavaScript: use 'eslint' or 'njsscan'
        - If Python: Use 'bandit' or 'safety'
        etc.
        """

    print(f"\n{'='*20} EVOLUTION PHASE: {state['specialization']} {'='*20}")
    print(f"🛠️ Current Tech Stack: {state['tech_stack']}")

    # We still bind the tools because the LLM might decide it needs ONE more
    # tool before it's done, but the conditional prompt discourages repeats.
    response = llm.bind_tools([install_and_develop_tool]).invoke(
        [SystemMessage(content=prompt)] + state["messages"]
    )
    print(f"DEBUG: Evolution decision: {response.content}")

    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"🏗️  [EVOLVING]: Installing tool '{tool_call['args'].get('tool_name')}'")
            print(f"    > Setup: {tool_call['args'].get('setup_command')}")
            print(f"    > Usage: {tool_call['args'].get('execution_command')}")

    print(f"{'='*25} EVOLUTION STEP COMPLETE {'='*25}\n")
    return {"messages": [response]}


def generalist_audit_phase(state: StemState):
    """The generalist agent exploring and auditing."""
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

    print(f"\n{'='*20} 🛡️  SPECIALIZED AUDIT: {state['specialization']} {'='*20}")
    evolved_tool_names = list(DEVELOPED_TOOLS.keys())
    print(f"🧰 Available Evolved Tools: {evolved_tool_names if evolved_tool_names else 'None (Manual Only)'}")

    # Bind only the basic vision tools
    tools = [list_directory_structure, read_file_content]
    response = llm.bind_tools(tools).invoke([SystemMessage(content=prompt)] + state["messages"])

    return {"messages": [response]}


def specialized_audit_phase(state: StemState):
    """The specialized agent auditing with custom tools."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = f"""
    You are a {state['specialization']} Code Auditor.
    Target Repository: {state['repo_name']}
    
    Your task:
    Perform a deep-dive audit of the repository '{state['repo_name']}'. You have evolved new tools (like {list(DEVELOPED_TOOLS.keys())}). You must execute these tools on the source code directory to gather empirical data.
    Do not rely on your memory. If you do not run your specialized tools, your audit is invalid.
    You can optionally also utilise the basic vision tools at your disposal ('list_directory_structure', 'read_file_content') if deemed necessary, but using the others is preferred.
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

    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"🚀 [EXECUTING]: {tool_call['name']} on target...")
    else:
        print("📊 [FINALIZING]: Agent is writing the final report.")
        # Check for issue count in final response
        count_match = re.search(r"TOTAL ISSUE COUNT:\s*(\d+)", response.content, re.I)
        if count_match:
            print(f"📈 AUDIT RESULT: {count_match.group(1)} issues identified.")

    print(f"{'='*25} AUDIT STEP COMPLETE {'='*25}\n")

    return {"messages": [response]}
