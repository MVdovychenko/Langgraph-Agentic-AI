import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_tavily import TavilySearch
from langchain_google_community import CalendarToolkit
from langchain_google_community.calendar.utils import (
    build_resource_service,
    get_google_credentials,
)


# Loads secrets and sets the working directory.
# If the API keys are missing, the app will stop immediately.
# You need to create a .env file and add your OPENAI_API_KEY and TAVILY_API_KEY there.
# The .env file is not included in the repo because it contains private credentials.

def bootstrap_secrets() -> None:
    """Load .env, chdir to /app/secrets if present, and ensure required keys exist."""
    load_dotenv()
    secrets_dir = Path("/app/secrets")
    if secrets_dir.exists():
        os.chdir(secrets_dir)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY (set it in .env or container env).")
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("Missing TAVILY_API_KEY (set it in .env or container env).")




# Sets up Google Calendar tools for the agent (create, update, search events).
# Uses CalendarToolkit to wrap the API instead of calling Google HTTP directly.
# Needs credentials.json (from Google Console) and token.json (created after first login).

def build_calendar_tools():
    """Return CalendarToolkit tools (requires credentials.json and token.json)."""
    credentials = get_google_credentials(
        token_file="token.json",
        scopes=["https://www.googleapis.com/auth/calendar.events"],
        client_secrets_file="credentials.json",
    )
    api_resource = build_resource_service(credentials=credentials)
    return CalendarToolkit(api_resource=api_resource).get_tools()




# Agents:

# Function builds the calendar agent with its prompt and tools that uses the provided 
# Google Calendar tools to create, update, search, or delete events. 
# The output is always forced into a JSON block (RESULT_JSON) so it can be parsed later. 
# Timezone is fixed to Europe/Berlin to keep results consistent.

def create_calendar_agent(tools):
    """
    Calendar work agent: DOES work using tools and outputs ONLY a fenced JSON block:

    ```json RESULT_JSON
    { "agent":"calendar","op":"create|update|delete|move|search|info|error",
      "ok":true|false,"data":<tool output>,"message":"...", "timezone":"Europe/Berlin" }
    ```
    """
    prompt = (
        "You are a Calendar work agent.\n\n"
        "TASK:\n"
        "- Use the provided tools to create, search, update, move, and delete events.\n"
        "- ALWAYS set timezone to 'Europe/Berlin'.\n"
        "- If date/time/calendar is missing, ask exactly one clarifying question via tool use if needed.\n\n"
        "OUTPUT CONTRACT (very important):\n"
        "- After completing the operation(s), respond ONLY with a single fenced JSON block labeled RESULT_JSON.\n"
        "- No other text. JSON must match the schema in the docstring."
    )
    model = init_chat_model("openai:gpt-4o-mini")
    return create_react_agent(model=model, tools=tools, prompt=prompt, name="calendar_agent")




# Function builds the research agent with its prompt and tools. 
# It uses TavilySearch to perform web searches and gather information. 
# The agent always outputs the results in a JSON block (RESULT_JSON) with a simple schema 
# that includes title, url, and snippet for each search result.

def create_research_agent():
    """
    Research work agent: DOES research via Tavily and outputs ONLY a fenced JSON block:

    ```json RESULT_JSON
    { "agent":"research","op":"search","ok":true|false,
      "data":[{"title":"...","url":"...","snippet":"..."}], "message":"..." }
    ```
    """
    prompt = (
        "You are a Research work agent.\n\n"
        "TASK:\n"
        "- Perform only research-related tasks using Tavily.\n\n"
        "OUTPUT CONTRACT (very important):\n"
        "- Respond ONLY with a single fenced JSON block labeled RESULT_JSON. No other text.\n"
        "- Normalize results to the schema in the docstring. If no hits, ok=false, data=[]."
    )
    model = init_chat_model("openai:gpt-4o-mini")
    web_search = TavilySearch(max_results=5)
    return create_react_agent(model=model, tools=[web_search], prompt=prompt, name="research_agent")




# Function builds the formatter agent. Agent does not call any tools. It follows specific rendering rules 
# based on the content of RESULT_JSON.
# The agent reads the most recent RESULT_JSON block from the conversation context, parses it, and
# produces a clean, user-friendly Markdown summary of the results.

def create_formatter_agent():
    """
    Formatter agent: DOES NOT call tools. It scans prior messages, finds the most recent
    fenced JSON block labeled RESULT_JSON, parses it, and produces user-friendly Markdown.

    - calendar/search: list events (summary, start, end, location, attendees)
    - calendar/create|update|move|delete: short confirmation with key fields
    - research/search: list up to 5 links: [Title](url) — snippet
    - fallback: if data is an array of objects, infer columns and show a simple table
    - If no RESULT_JSON present: explain that no tool output was found to format
    """
    prompt = (
        "You are a Formatter agent.\n\n"
        "TASK:\n"
        "- Read the conversation so far, find the most recent fenced JSON block labeled RESULT_JSON,\n"
        "  parse it, and produce a clean, user-friendly Markdown summary of the results.\n"
        "- Do NOT call any tools.\n\n"
        "RENDERING RULES:\n"
        "- If RESULT_JSON.agent == 'calendar':\n"
        "   * op=='search': bulleted list of events (summary, start, end, location, attendees).\n"
        "   * op in ['create','update','move','delete']: concise confirmation with key fields.\n"
        "   * if ok==false: concise error or 'no results'.\n"
        "- If RESULT_JSON.agent == 'research':\n"
        "   * ok==true: up to 5 items as: [Title](url) — snippet.\n"
        "   * ok==false: 'no high-quality sources found'.\n"
        "- Fallback: if data is an array of objects, infer up to 6 columns and render a simple table.\n"
        "- Keep it concise. Use Markdown only. No JSON or code blocks in the final output."
    )
    model = init_chat_model("openai:gpt-4o-mini")
    return create_react_agent(model=model, tools=[], prompt=prompt, name="formatter_agent")



# Function builds the supervisor agent. 
# The supervisor decides which work agent (research or calendar) should handle the user’s request. 
# It always picks exactly one agent and does not add its own answer. 
# The chosen agent then produces the RESULT_JSON, which will be formatted later.

def create_supervisor_runnable(research_agent, calendar_agent):
    """
    Supervisor policy:
    - Choose exactly one WORK agent (research_agent or calendar_agent).
    - After the chosen agent returns, STOP. Do not add your own user-facing message.
    - The work agent must output a RESULT_JSON block which the next node will format.
    """
    sup_prompt = (
        "You are a supervisor managing two WORK agents:\n"
        "- research_agent: research tasks via Tavily.\n"
        "- calendar_agent: calendar CRUD via Google Calendar tools.\n\n"
        "POLICY:\n"
        "1) Pick exactly one WORK agent for the user's request.\n"
        "2) After the WORK agent finishes, do NOT write any additional message. End immediately.\n"
        "3) The WORK agent's reply MUST be a RESULT_JSON block. Do not format it yourself."
    )
    return create_supervisor(
        model=init_chat_model("openai:gpt-4o-mini"),
        agents=[research_agent, calendar_agent],
        prompt=sup_prompt,
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile()



# Function creates the overall graph that connects all agents. 
# The order is: START → supervisor → formatter_agent → END. 
# This means the supervisor picks the right agent, and the formatter 
# then turns the JSON output into readable text for the user.

def build_parent_graph(supervisor_runnable, formatter_agent):
    """
    Parent graph topology:
        START → supervisor → formatter_agent → END
    """
    builder = StateGraph(MessagesState)
    builder.add_node("supervisor", supervisor_runnable)
    builder.add_node("formatter_agent", formatter_agent)
    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "formatter_agent")
    builder.add_edge("formatter_agent", END)
    return builder.compile()



# Function runs the whole agent graph once with the given user text. 
# It sends the user’s message into the system, waits for the result, 
# and then returns only the final formatted answer (from the formatter agent).

def run_graph(app, user_text: str) -> str:
    """Invoke the graph once and return the final assistant text (from formatter)."""
    result = app.invoke({"messages": [HumanMessage(content=user_text)]})
    msgs = result.get("messages", [])
    if not msgs:
        return ""
    last = msgs[-1]
    content = getattr(last, "content", "")
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content)



# The main entry point of the app. It sets up the Streamlit interface (chat window), 
# builds all agents and the supervisor, and then runs the agent graph for each user input. 
# Messages are stored in session_state so the chat history is visible.

def main():
    st.set_page_config(page_title="AI Agent System", page_icon="🤖", layout="wide")
    bootstrap_secrets()

    # keep messages and other data in the session
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("cache", {})
    st.session_state.setdefault("context", None)

    # build all agents and the supervisor
    calendar_tools = build_calendar_tools()
    calendar_agent = create_calendar_agent(calendar_tools)
    research_agent = create_research_agent()
    formatter_agent = create_formatter_agent()

    supervisor_node = create_supervisor_runnable(research_agent, calendar_agent)
    app = build_parent_graph(supervisor_node, formatter_agent)

     # create diagrams of the agent graph
    png_parent = app.get_graph().draw_mermaid_png()
    png_super  = supervisor_node.get_graph().draw_mermaid_png()

    #uncomment to generate graph about architecture
    # with st.expander("🗺️ Graphs"):
    #     st.image(png_parent, caption="Parent Graph", use_column_width=True)
    #     st.image(png_super,  caption="Supervisor Subgraph", use_column_width=True)


    # chat interface
    st.markdown("### 🤖 Ask me something...")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_text := st.chat_input("Ask something..."):
        with st.chat_message("user"):
            st.markdown(user_text)
        st.session_state.messages.append({"role": "user", "content": user_text})

        with st.chat_message("assistant"):
            reply = run_graph(app, user_text)
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()
