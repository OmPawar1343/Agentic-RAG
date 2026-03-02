# agent_langgraph.py
"""
LangGraph Agent for CSV RAG Privacy Bot
Enhanced with Safe Tools + RBAC Protection
"""

import os
import sys

# ============================================
# 🔑 PUT YOUR API KEY HERE
# ============================================
GROQ_API_KEY = 
# ============================================

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

from typing import Annotated, List, TypedDict, Literal, Sequence
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langchain_core.messages import (
    BaseMessage, 
    HumanMessage, 
    SystemMessage, 
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import tool

import rag

# ==================== Configuration ====================

GROQ_MODEL = "llama-3.1-8b-instant"
MAX_TOOL_CALLS = 5  # Increased for more tools

print(f"🔧 Initializing LangGraph Agent...")

# ==================== LLM Initialization ====================

llm = None
LLM_DISPLAY_NAME = "Not initialized"


def init_groq_llm() -> bool:
    global llm, LLM_DISPLAY_NAME
    
    if not GROQ_API_KEY or "xxxx" in GROQ_API_KEY:
        print("❌ Please set your Groq API key!")
        return False
    
    try:
        from langchain_groq import ChatGroq
        
        print(f"🔗 Testing Groq API...")
        
        llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0,
            max_tokens=500,
        )
        
        # Quick test
        llm.invoke([HumanMessage(content="Hi")])
        
        LLM_DISPLAY_NAME = f"Groq ({GROQ_MODEL})"
        print(f"✅ LLM: {LLM_DISPLAY_NAME} - Connected!")
        return True
        
    except ImportError:
        print("❌ Run: pip install langchain-groq")
        return False
    except Exception as e:
        print(f"❌ Groq error: {e}")
        return False


if not init_groq_llm():
    print("\n❌ Fix your API key and try again")
    sys.exit(1)


# ==================== Helper Functions ====================

def _clean_rag_result(result: str) -> str:
    """
    Extract and clean main content from RAG result.
    Removes Privacy Meter and formats table data nicely.
    """
    # Remove Privacy Meter section
    parts = result.split("## Privacy Meter")
    main_content = parts[0].strip()
    
    # Extract LLM Summary
    summary = ""
    if "## LLM Summary" in main_content:
        summary_parts = main_content.split("## LLM Summary")
        main_content = summary_parts[0].strip()
        if len(summary_parts) > 1:
            summary = summary_parts[1].strip()
    
    # Parse table into readable format
    lines = []
    for line in main_content.split("\n"):
        line = line.strip()
        if line and not line.startswith("|---"):
            if line.startswith("##"):
                lines.append(line.replace("##", "").strip() + ":")
            elif line.startswith("|"):
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if len(cells) >= 2 and cells[0] not in ("Column", "Metric"):
                    lines.append(f"  {cells[0]}: {cells[1]}")
            else:
                lines.append(line)
    
    result_text = "\n".join(lines) if lines else ""
    
    # Add summary if available and valid
    if summary and "error" not in summary.lower() and len(summary) > 10:
        result_text += f"\n\nSummary: {summary}"
    
    return result_text if result_text else "No data found."


def _require_auth() -> str:
    """Check if user is authenticated."""
    if not rag.CURRENT_USER:
        return "Error: You must be logged in to use this feature. Please login first."
    return None


# ==================== SAFE TOOLS ====================

# ----- Tool 1: General Query (Primary Tool) -----
@tool
def query_employee_database(question: str) -> str:
    """
    Query the employee database with any question about employees.
    This is the main tool for all employee-related queries.
    RBAC and privacy controls are automatically applied.
    
    Args:
        question: Natural language question about employees
    
    Returns:
        Employee information based on user's access level
    
    Examples:
        - "List all employees"
        - "Who works in Sales department?"
        - "What is John Smith's job title?"
    """
    auth_error = _require_auth()
    if auth_error:
        return auth_error
    
    try:
        result = rag.rag_query(question)
        return _clean_rag_result(result)
    except Exception as e:
        return f"Error querying database: {str(e)}"


# ----- Tool 2: Search by Name -----
@tool
def search_employee_by_name(employee_name: str) -> str:
    """
    Search for a specific employee by their name.
    Returns employee details based on your access level.
    
    Args:
        employee_name: Full name or partial name of the employee
    
    Returns:
        Employee information if found
    
    Examples:
        - search_employee_by_name("John Smith")
        - search_employee_by_name("Sarah")
    """
    auth_error = _require_auth()
    if auth_error:
        return auth_error
    
    if not employee_name or len(employee_name.strip()) < 2:
        return "Please provide a valid employee name (at least 2 characters)."
    
    try:
        result = rag.rag_query(f"Find employee named {employee_name}")
        return _clean_rag_result(result)
    except Exception as e:
        return f"Error searching for employee: {str(e)}"


# ----- Tool 3: Search by Employee ID -----
@tool
def search_employee_by_id(employee_id: str) -> str:
    """
    Search for a specific employee by their Employee ID.
    Returns employee details based on your access level.
    
    Args:
        employee_id: The employee's ID number
    
    Returns:
        Employee information if found
    
    Examples:
        - search_employee_by_id("1001")
        - search_employee_by_id("E12345")
    """
    auth_error = _require_auth()
    if auth_error:
        return auth_error
    
    if not employee_id or len(employee_id.strip()) < 1:
        return "Please provide a valid employee ID."
    
    try:
        result = rag.rag_query(f"Get details for employee ID {employee_id}")
        return _clean_rag_result(result)
    except Exception as e:
        return f"Error searching for employee: {str(e)}"


# ----- Tool 4: Get Specific Field -----
@tool
def get_employee_field(employee_name_or_id: str, field_name: str) -> str:
    """
    Get a specific field/attribute for an employee.
    Only returns fields you have permission to access.
    
    Args:
        employee_name_or_id: Employee name or ID
        field_name: The specific field to retrieve (e.g., "Department", "JobTitle", "Location")
    
    Returns:
        The requested field value if accessible
    
    Examples:
        - get_employee_field("John Smith", "Department")
        - get_employee_field("1001", "JobTitle")
    """
    auth_error = _require_auth()
    if auth_error:
        return auth_error
    
    if not employee_name_or_id or not field_name:
        return "Please provide both employee name/ID and field name."
    
    try:
        result = rag.rag_query(f"What is the {field_name} of {employee_name_or_id}?")
        return _clean_rag_result(result)
    except Exception as e:
        return f"Error retrieving field: {str(e)}"


# ----- Tool 5: List Employees by Department -----
@tool
def list_employees_by_department(department_name: str) -> str:
    """
    List all employees in a specific department.
    Results are filtered based on your access level.
    
    Args:
        department_name: Name of the department (e.g., "Sales", "Engineering", "HR")
    
    Returns:
        List of employees in that department
    
    Examples:
        - list_employees_by_department("Sales")
        - list_employees_by_department("Engineering")
    """
    auth_error = _require_auth()
    if auth_error:
        return auth_error
    
    if not department_name:
        return "Please provide a department name."
    
    try:
        result = rag.rag_query(f"List all employees in {department_name} department")
        return _clean_rag_result(result)
    except Exception as e:
        return f"Error listing employees: {str(e)}"


# ----- Tool 6: Count Employees -----
@tool
def count_employees(filter_criteria: str = "") -> str:
    """
    Count employees, optionally with filter criteria.
    
    Args:
        filter_criteria: Optional filter (e.g., "in Sales", "with job title Manager")
                        Leave empty for total count.
    
    Returns:
        Count of employees matching criteria
    
    Examples:
        - count_employees()  # Total count
        - count_employees("in Sales department")
        - count_employees("with Manager in their title")
    """
    auth_error = _require_auth()
    if auth_error:
        return auth_error
    
    try:
        if filter_criteria:
            query = f"How many employees {filter_criteria}?"
        else:
            query = "How many employees are there in total?"
        
        result = rag.rag_query(query)
        return _clean_rag_result(result)
    except Exception as e:
        return f"Error counting employees: {str(e)}"


# ----- Tool 7: Get My Information (Self-Service) -----
@tool
def get_my_information(specific_field: str = "") -> str:
    """
    Get your own employee information (self-service).
    You have full access to your own data.
    
    Args:
        specific_field: Optional - specific field to retrieve.
                       Leave empty for all your details.
    
    Returns:
        Your employee information
    
    Examples:
        - get_my_information()  # All your details
        - get_my_information("Department")
        - get_my_information("JobTitle")
    """
    auth_error = _require_auth()
    if auth_error:
        return auth_error
    
    user = rag.CURRENT_USER
    user_name = user.get("FullName", "")
    
    if not user_name:
        return "Your user profile doesn't have a name configured."
    
    try:
        if specific_field:
            query = f"What is my {specific_field}?"
        else:
            query = "Show all my details"
        
        result = rag.rag_query(query)
        return _clean_rag_result(result)
    except Exception as e:
        return f"Error retrieving your information: {str(e)}"


# ----- Tool 8: Get Current User Info -----
@tool
def get_current_user_info() -> str:
    """
    Get information about the currently logged-in user.
    Shows your login status, role, and access level.
    
    Returns:
        Current user's login information and role
    """
    user = rag.CURRENT_USER or {}
    
    if not user:
        return "No user logged in. Please use the 'login' command to authenticate."
    
    pii_status = "🔓 Unlocked" if user.get("pii_unlocked") else "🔒 Locked"
    
    return f"""Current User Information:
  Name: {user.get('FullName', 'Unknown')}
  Role: {user.get('role', 'Unknown')}
  Employee ID: {user.get('EmpID', 'N/A')}
  PII Access: {pii_status}"""


# ----- Tool 9: Get Database Schema -----
@tool
def get_database_schema() -> str:
    """
    Get the available fields/columns in the employee database.
    Shows which fields exist and which are accessible to you.
    
    Returns:
        List of database fields and your access level
    """
    auth_error = _require_auth()
    if auth_error:
        return auth_error
    
    user = rag.CURRENT_USER or {}
    role = user.get("role", "user")
    
    all_cols = rag.ALL_COLUMNS
    sensitive_cols = rag.SENSITIVE_COLS
    non_sensitive_cols = rag.NON_SENSITIVE_COLS
    
    if role == "admin":
        accessible = all_cols
        restricted = []
    else:
        accessible = non_sensitive_cols
        restricted = list(sensitive_cols)
    
    result = f"""Database Schema:

📊 Total Fields: {len(all_cols)}

✅ Accessible Fields ({len(accessible)}):
  {', '.join(accessible[:15])}{'...' if len(accessible) > 15 else ''}

🔒 Restricted Fields ({len(restricted)}):
  {', '.join(restricted[:10]) if restricted else 'None (Admin access)'}{'...' if len(restricted) > 10 else ''}

Your Role: {role.upper()}"""
    
    return result


# ----- Tool 10: Check Access Permissions -----
@tool
def check_access_permissions() -> str:
    """
    Check what data and operations you have access to.
    Shows your permission level and any restrictions.
    
    Returns:
        Detailed description of your access permissions
    """
    auth_error = _require_auth()
    if auth_error:
        return auth_error
    
    user = rag.CURRENT_USER or {}
    role = user.get("role", "user")
    pii_unlocked = user.get("pii_unlocked", False)
    
    if role == "admin":
        return f"""🔑 ACCESS LEVEL: ADMINISTRATOR

Permissions:
  ✅ View all employee records
  ✅ Access all database fields
  ✅ View sensitive fields (PII): {'UNLOCKED' if pii_unlocked else 'LOCKED - use "unlock" command'}
  ✅ Query any employee data
  ✅ Bypass some safety guards

Sensitive Fields Access:
  {'🔓 UNLOCKED - Full PII visible' if pii_unlocked else '🔒 LOCKED - PII is masked. Use "unlock" to reveal.'}

Available Commands:
  - unlock: Reveal sensitive PII data
  - lock: Hide sensitive PII data"""
    else:
        return f"""👤 ACCESS LEVEL: STANDARD USER

Permissions:
  ✅ View your OWN employee record
  ✅ Access non-sensitive fields
  ❌ Cannot view other employees' sensitive data
  ❌ Cannot access: Email, Phone, Salary, DOB, Address, etc.

Your Data:
  - You have FULL access to your own record
  - Name: {user.get('FullName', 'N/A')}
  - Employee ID: {user.get('EmpID', 'N/A')}

To view your information:
  - "Show my details"
  - "What is my department?"
  - Use get_my_information() tool"""


# ----- Tool 11: Get Privacy Status -----
@tool
def get_privacy_status() -> str:
    """
    Get the current privacy and security status of the system.
    Shows active guards, protection levels, and audit status.
    
    Returns:
        Privacy and security configuration status
    """
    user = rag.CURRENT_USER or {}
    role = user.get("role", "guest")
    
    # Check which guards are active
    guards_status = []
    
    try:
        from safety_orchestrator import (
            USE_PRESIDIO, USE_LLM_GUARD, USE_GUARDRAILS, USE_REGEX,
            PRESIDIO, LLM_GUARD_INPUT, LLM_GUARD_OUTPUT, GUARDRAILS_GUARD
        )
        
        guards_status.append(f"  Presidio PII Detection: {'✅ Active' if USE_PRESIDIO and PRESIDIO else '❌ Inactive'}")
        guards_status.append(f"  LLM Guard (Input): {'✅ Active' if USE_LLM_GUARD and LLM_GUARD_INPUT else '❌ Inactive'}")
        guards_status.append(f"  LLM Guard (Output): {'✅ Active' if USE_LLM_GUARD and LLM_GUARD_OUTPUT else '❌ Inactive'}")
        guards_status.append(f"  Guardrails Validation: {'✅ Active' if USE_GUARDRAILS and GUARDRAILS_GUARD else '❌ Inactive'}")
        guards_status.append(f"  Regex PII Detection: {'✅ Active' if USE_REGEX else '❌ Inactive'}")
    except ImportError:
        guards_status.append("  (Guard status unavailable)")
    
    privacy_meter = "✅ Enabled" if rag.PRIVACY_METER else "❌ Disabled"
    
    return f"""🛡️ PRIVACY & SECURITY STATUS

User: {user.get('FullName', 'Not logged in')} ({role})

Security Guards:
{chr(10).join(guards_status)}

Privacy Controls:
  Privacy Meter: {privacy_meter}
  Sensitive Columns Protected: {len(rag.SENSITIVE_COLS)}
  RBAC Enforcement: ✅ Active
  Self-Only Mode: {'✅ Enabled' if rag.ENFORCE_SELF_ONLY else '❌ Disabled'}

Data Protection:
  PII Redaction: ✅ Active
  Denylist Patterns: ✅ Active
  Injection Prevention: ✅ Active"""


# ==================== TOOLS LIST ====================

# All safe tools that respect RBAC and privacy
tools = [
    # Primary query tool
    query_employee_database,
    
    # Search tools
    search_employee_by_name,
    search_employee_by_id,
    
    # Specific data tools
    get_employee_field,
    list_employees_by_department,
    count_employees,
    
    # Self-service tools
    get_my_information,
    
    # Info/status tools
    get_current_user_info,
    get_database_schema,
    check_access_permissions,
    get_privacy_status,
]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)
print(f"✅ Tools ({len(tools)}): {[t.name for t in tools]}")


# ==================== Agent State ====================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_count: int
    final_answer: str


# ==================== System Prompt ====================

SYSTEM_PROMPT = """You are a helpful HR assistant with access to an employee database.

AVAILABLE TOOLS:
1. query_employee_database - General queries about employees
2. search_employee_by_name - Find employee by name
3. search_employee_by_id - Find employee by ID
4. get_employee_field - Get specific field for an employee
5. list_employees_by_department - List employees in a department
6. count_employees - Count employees (with optional filter)
7. get_my_information - Get YOUR OWN employee info
8. get_current_user_info - Check who is logged in
9. get_database_schema - See available fields
10. check_access_permissions - Check your access level
11. get_privacy_status - Check security settings

RULES:
1. Use the most specific tool for the task
2. For personal data requests, use get_my_information
3. After getting tool results, summarize in 2-3 sentences
4. Do NOT make up data - only report what tools return
5. Respect privacy - don't try to bypass access controls
6. If access is denied, explain politely

IMPORTANT: After receiving tool results, respond naturally. Do NOT call tools repeatedly."""


# ==================== Agent Nodes ====================

def agent_node(state: AgentState) -> dict:
    """Agent decides what to do."""
    messages = list(state["messages"])
    tool_count = state.get("tool_call_count", 0)
    
    # Add system prompt if needed
    if not messages or not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
    
    # If we've called tools too many times, force a summary
    if tool_count >= MAX_TOOL_CALLS:
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                return {
                    "messages": [AIMessage(content=f"Based on the database: {msg.content[:500]}")],
                    "tool_call_count": tool_count
                }
        return {
            "messages": [AIMessage(content="I couldn't find the information you requested.")],
            "tool_call_count": tool_count
        }
    
    try:
        response = llm_with_tools.invoke(messages)
        return {"messages": [response], "tool_call_count": tool_count}
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Error: {str(e)}")],
            "tool_call_count": tool_count
        }


def tool_node(state: AgentState) -> dict:
    """Execute tools with safety checks."""
    messages = state["messages"]
    last_message = messages[-1]
    tool_count = state.get("tool_call_count", 0)
    
    tool_results = []
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", f"call_{tool_name}")
            
            result = f"Unknown tool: {tool_name}"
            
            # Find and execute the tool
            for t in tools:
                if t.name == tool_name:
                    try:
                        result = t.invoke(tool_args)
                    except Exception as e:
                        result = f"Tool error: {str(e)}"
                    break
            
            tool_results.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_id,
                name=tool_name,
            ))
            tool_count += 1
    
    return {"messages": tool_results, "tool_call_count": tool_count}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Decide next step."""
    messages = state["messages"]
    last_message = messages[-1]
    tool_count = state.get("tool_call_count", 0)
    
    # Force end if too many tool calls
    if tool_count >= MAX_TOOL_CALLS:
        return "end"
    
    # Check if LLM wants to call tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "end"


def format_response_node(state: AgentState) -> dict:
    """Format final response."""
    messages = state["messages"]
    
    # Try to find the last AI message (not a tool call)
    final_answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                final_answer = msg.content
                break
    
    # If no AI response, use the last tool result
    if not final_answer:
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                final_answer = f"Here's what I found:\n{msg.content[:500]}"
                break
    
    if not final_answer:
        final_answer = "I couldn't generate a response."
    
    return {"final_answer": final_answer}


# ==================== Build Graph ====================

print("🔨 Building agent graph...")
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("format", format_response_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": "format"})
workflow.add_edge("tools", "agent")
workflow.add_edge("format", END)
agent_graph = workflow.compile()
print("✅ Agent ready!")


# ==================== Runner ====================

def run_agent(question: str, verbose: bool = False) -> dict:
    """Run agent with recursion limit."""
    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question)
        ],
        "tool_call_count": 0,
        "final_answer": "",
    }
    
    if verbose:
        print(f"\n📝 Question: {question}")
        print("-" * 50)
    
    try:
        config = {"recursion_limit": 50}
        final_state = agent_graph.invoke(initial_state, config=config)
        
        if verbose:
            for msg in final_state.get("messages", []):
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"🔧 Tool: {tc.get('name')}")
                elif isinstance(msg, ToolMessage):
                    content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                    print(f"📤 Result: {content}")
            print("-" * 50)
        
        return {
            "answer": final_state.get("final_answer", "No response"),
            "messages": final_state.get("messages", [])
        }
        
    except Exception as e:
        error_msg = str(e)
        if "recursion" in error_msg.lower():
            try:
                result = rag.rag_query(question)
                if "## LLM Summary" in result:
                    parts = result.split("## LLM Summary")
                    if len(parts) > 1:
                        return {"answer": parts[1].split("##")[0].strip(), "messages": []}
                return {"answer": result.split("##")[0].strip()[:500], "messages": []}
            except:
                pass
        return {"answer": f"Error: {error_msg}", "messages": []}


def run_simple_query(question: str) -> str:
    """Direct RAG query (no agent)."""
    try:
        result = rag.rag_query(question)
        if "## LLM Summary" in result:
            parts = result.split("## LLM Summary")
            if len(parts) > 1:
                return parts[1].split("##")[0].strip()
        return result.split("## Privacy Meter")[0].strip()
    except Exception as e:
        return f"Error: {str(e)}"


# ==================== CLI ====================

def print_welcome():
    print("\n" + "=" * 60)
    print("🤖 CSV RAG Agent - Enhanced with Safe Tools")
    print("=" * 60)
    print(f"📡 LLM: {LLM_DISPLAY_NAME}")
    print(f"🔧 Tools: {len(tools)} available")
    print("-" * 60)
    print("Commands:")
    print("  login/switch - Change user")
    print("  whoami       - Show current user")
    print("  unlock/lock  - Toggle PII access (admin)")
    print("  verbose      - Toggle verbose mode")
    print("  simple       - Toggle simple mode (no agent)")
    print("  tools        - List available tools")
    print("  help/?       - Show this help")
    print("  exit/quit    - Exit")
    print("=" * 60)


def print_tools():
    """Print available tools with descriptions."""
    print("\n🔧 Available Tools:")
    print("-" * 50)
    for i, t in enumerate(tools, 1):
        desc = t.description.split('\n')[0] if t.description else "No description"
        print(f"  {i:2}. {t.name}")
        print(f"      {desc[:60]}...")
    print("-" * 50)


def handle_command(cmd: str, verbose: bool, simple: bool) -> tuple:
    cmd = cmd.lower().strip()
    
    if cmd in ("exit", "quit", "q"):
        print("\n👋 Goodbye!")
        return True, True, verbose, simple
    
    if cmd in ("login", "switch"):
        try:
            rag.CURRENT_USER = rag.authenticate()
            print(f"✅ Logged in: {rag.CURRENT_USER.get('FullName')}")
        except:
            pass
        return True, False, verbose, simple
    
    if cmd == "unlock":
        if rag.CURRENT_USER and rag.CURRENT_USER.get("role") == "admin":
            from getpass import getpass
            pw = getpass("Password: ")
            expected = getattr(rag, 'ADMIN_PII_UNLOCK_PASSWORD', None) or rag.CURRENT_USER.get("password")
            if pw == expected:
                rag.CURRENT_USER["pii_unlocked"] = True
                print("🔓 PII Unlocked")
            else:
                print("❌ Wrong password")
        else:
            print("❌ Admin access required")
        return True, False, verbose, simple
    
    if cmd == "lock":
        if rag.CURRENT_USER:
            rag.CURRENT_USER["pii_unlocked"] = False
            print("🔒 PII Locked")
        return True, False, verbose, simple
    
    if cmd in ("whoami", "who"):
        u = rag.CURRENT_USER or {}
        pii = "🔓" if u.get("pii_unlocked") else "🔒"
        print(f"👤 {u.get('FullName', 'Guest')} | Role: {u.get('role', '-')} | PII: {pii}")
        return True, False, verbose, simple
    
    if cmd == "verbose":
        print(f"📝 Verbose: {'OFF → ON' if not verbose else 'ON → OFF'}")
        return True, False, not verbose, simple
    
    if cmd == "simple":
        print(f"🔄 Mode: {'Agent → Simple' if not simple else 'Simple → Agent'}")
        return True, False, verbose, not simple
    
    if cmd == "tools":
        print_tools()
        return True, False, verbose, simple
    
    if cmd in ("help", "?"):
        print_welcome()
        return True, False, verbose, simple
    
    return False, False, verbose, simple


def main():
    print_welcome()
    
    try:
        rag.CURRENT_USER = rag.authenticate()
        print(f"\n✅ Welcome, {rag.CURRENT_USER.get('FullName')} ({rag.CURRENT_USER.get('role')})")
    except:
        print("❌ Authentication failed")
        return
    
    verbose, simple = False, False
    
    print("\n💡 Example queries:")
    print("  - 'List all employees'")
    print("  - 'Show my details'")
    print("  - 'What is John Smith's department?'")
    print("  - 'How many employees in Sales?'")
    print("  - Type 'tools' to see all available tools\n")
    
    while True:
        try:
            u = rag.CURRENT_USER or {}
            icon = "🔑" if u.get("role") == "admin" else "👤"
            lock = "🔓" if u.get("pii_unlocked") else "🔒"
            mode = "[Simple]" if simple else "[Agent]"
            
            question = input(f"\n{icon}{lock} {mode} > ").strip()
            
            if not question:
                continue
            
            was_cmd, should_exit, verbose, simple = handle_command(question, verbose, simple)
            if should_exit:
                break
            if was_cmd:
                continue
            
            print("\n🤔 Processing...")
            
            if simple:
                answer = run_simple_query(question)
            else:
                result = run_agent(question, verbose=verbose)
                answer = result["answer"]
            
            print(f"\n🤖 {answer}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Type 'exit' to quit")
        except EOFError:
            break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("🧪 Running tests...")
        rag.CURRENT_USER = {"FullName": "Test Admin", "role": "admin", "EmpID": "0"}
        
        print("\n--- Test 1: List employees ---")
        result = run_agent("list employees", verbose=True)
        print(f"Answer: {result['answer'][:200]}...")
        
        print("\n--- Test 2: Check permissions ---")
        result = run_agent("check my access permissions", verbose=True)
        print(f"Answer: {result['answer'][:200]}...")
        
    else:
        main()