# Browser Based interface using Streamlit UI.
# Create function get_agent() -> remove memory object and directly created in get_agent() ->
# streamlite header, user input, and displaying user/Ai messages i browser.

from dotenv import load_dotenv
load_dotenv()

# llm, tools, memory, agent,

# llm
# from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
# tools
from langchain_community.utilities import SQLDatabase # used to connect with db
from langchain_community.agent_toolkits import SQLDatabaseToolkit
# memory
from langgraph.checkpoint.memory import InMemorySaver
# agent
from langchain.agents import create_agent
# streamlit UI
import streamlit as st


# creating the database and table, we are using sqllite.
db = SQLDatabase.from_uri("sqlite:///my_tasks.db")
db.run("""
CREATE TABLE IF NOT EXISTS my_tasks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    descriptions TEXT,
    status TEXT CHECK (status IN ('pending', 'in_progress', 'complete')) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# model = ChatGroq(model="openai/gpt-oss-20b")
model = ChatAnthropic(model="claude-opus-5")
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
system_prompt = """
You are a task management assistant that interacts with a SQL database containing a 'my_tasks' table. 

TASK RULES:
1. Limit SELECT queries to 10 results max with ORDER BY created_at DESC
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If the user requests a list of tasks, present the output in a structured table format to ensure a clean and organized display in the browser."

CRUD OPERATIONS:
    CREATE: INSERT INTO my_tasks(title, descriptions, status)
    READ: SELECT * FROM my_tasks WHERE ... LIMIT 10
    UPDATE: UPDATE my_tasks SET status=? WHERE id=? OR title=?
    DELETE: DELETE FROM my_tasks WHERE id=? OR title=?

Table schema: id, title, descriptions, status(pending/in_progress/completed), created_at.
"""

# We have created following function b/c each time streamlit run it refreash the all code
# As a result the memory of the session lost, thats why we created a function and put a decorator.

@st.cache_resource
def get_agent():
    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=InMemorySaver(),
        system_prompt=system_prompt
    )
    return agent

agent = get_agent()

st.subheader("AI Powered Task Manger by Afraz.")

prompt = st.chat_input("I handle your tasks...")
if prompt:
    # user message part in browser
    st.chat_message("user").markdown(prompt)

    # AI message part in browser
    with st.chat_message("ai"):
        with st.spinner("processing..."):
            res = agent.invoke(
                {"messages":[{"role": "user", "content":prompt}]},
                {"configurable": {"thread_id":"1"}}
            )
            answer = res["messages"][-1].content
            st.markdown(answer)