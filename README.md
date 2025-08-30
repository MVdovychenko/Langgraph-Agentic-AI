# AI Agent System

This project is part of the assignment "Retrieval-Augmented Generation (RAG) und Agentensysteme".  
It implements a small multi-agent application using LangChain, LangGraph and Streamlit.  
The system can do two main tasks:
- manage events in Google Calendar
- search the web for information

The results are always returned in a structured JSON format and then converted into user-friendly Markdown.

---

## Setup

1. Clone the repository.
2. Create `.env` file in the project root and add:
   ```
   OPENAI_API_KEY=your-key
   TAVILY_API_KEY=your-key
   ```
3. In Google Cloud Console create OAuth credentials ("Web app") and download them as `credentials.json`.  
   Place the file into the `secrets/` folder.
4. Run `oauth_bootstrap.py` once. This opens a browser window and asks for Google login.  
   After the first login, a `token.json` is created in the same folder.
5. Start the app with Docker:
   ```
   docker-compose up
   ```

---

## Implemented Agent Components

From the list of six possible components, the following four were implemented:

### 1. Structured Output (JSON)
- **Where:** `create_calendar_agent`, `create_research_agent`
- Both agents are instructed to output a `RESULT_JSON` block only.  
- Example schema for calendar agent:
  ```json
  { "agent":"calendar","op":"create","ok":true,"data":{...},"message":"...", "timezone":"Europe/Berlin" }
  ```


### 2. Tool Usage
- **Where:**  
  - `build_calendar_tools` → provides Google Calendar tools (create, update, search, delete events).  
  - `create_research_agent` → uses TavilySearch to get search results from the web.
- Instead of writing raw HTTP requests, the agents call these ready-made tools.

### 3. Routing
- **Where:** `create_supervisor_runnable`
- The supervisor agent decides which subagent should handle the request:
  - if it’s a research question → research_agent  
  - if it’s about calendar → calendar_agent  

### 4. Orchestration
- **Where:** `build_parent_graph`
- The overall workflow is organized as a LangGraph:
  ```
  START → supervisor → formatter_agent → END
  ```
- The supervisor delegates the work, and the formatter agent turns the JSON into Markdown for the user.

---

## Streamlit Interface

The system is wrapped in a small Streamlit app (`main()` in `app.py`).  
It provides a simple chat interface:
- user enters a question,
- system decides which agent to use,
- formatter displays the final Markdown response in the chat.

---

## Notes

- `.env`, `credentials.json`, and `token.json` are **not included** in the repo for security reasons.  
- The reviewer needs to create these files locally to test the full functionality. check Google documentation for the token.json and credentials.json.
- oauth_bootstrap.py should be executed first time to generate the google token. then the app should work!
- you can simply run the script outside the container using simply "oauth_bootstrap.py".
