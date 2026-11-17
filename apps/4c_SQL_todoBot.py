# Preserving the Memory
# In addition to "4b" file -> Created an [] of messages in session state -> append the user/ai messages to messages[]
# -> 

from dotenv import load_dotenv
load_dotenv()

import sqlite3
from typing import Literal
from uuid import uuid4

from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import streamlit as st


def run_db(statement, parameters=(), *, fetch=False):
    connection = sqlite3.connect("my_tasks.db", timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            cursor = connection.execute(statement, parameters)
            if fetch:
                return [dict(row) for row in cursor.fetchall()]
            return cursor.rowcount
    finally:
        connection.close()


run_db("""
CREATE TABLE IF NOT EXISTS my_tasks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    descriptions TEXT,
    status TEXT CHECK (status IN ('pending', 'in_progress', 'complete')) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# model = ChatGroq(model="openai/gpt-oss-20b")
model = ChatAnthropic(
    model="claude-opus-5",
    disable_streaming="tool_calling",
)
system_prompt = """
You are a task management assistant. Use only the provided task tools; do not write SQL.

TASK RULES:
1. Add each requested task once; do not retry a successful tool call.
2. Use list_tasks to show up to 10 newest tasks.
3. When listing tasks, present them in a structured table.

Use only add_task, list_tasks, update_task, and delete_task for CRUD operations.
Use add_task once per requested task; never retry a successful write.
Use list_tasks to read tasks; it returns up to 10 newest items.
For updates, identify the task by ID and use update_task.
For deletions, identify the task by ID and use delete_task.
Only use the exact statuses pending, in_progress, and complete.

Stored task fields: id, title, descriptions, status, and created_at.
"""

def build_tools():
    added_titles = set()

    @tool
    def add_task(title: str, description: str = "") -> str:
        """Add one task with a title and optional description."""
        normalized_title = title.strip().casefold()
        if normalized_title in added_titles:
            return f"A task titled '{title}' was already added for this request; no duplicate was created."

        run_db(
            "INSERT INTO my_tasks (title, descriptions) VALUES (:title, :description)",
            parameters={"title": title.strip(), "description": description.strip()},
        )
        added_titles.add(normalized_title)
        return f"Task added: {title.strip()}"

    @tool
    def list_tasks() -> str:
        """List up to 10 most recently created tasks."""
        rows = run_db(
            "SELECT id, title, descriptions, status, created_at "
            "FROM my_tasks ORDER BY created_at DESC, id DESC LIMIT 10",
            fetch=True,
        )
        if not rows:
            return "No tasks found."
        return "\n".join(
            f"ID: {row['id']} | {row['title']} | {row['status']} | "
            f"{row['descriptions'] or ''} | {row['created_at']}"
            for row in rows
        )

    @tool
    def update_task(task_id: int, status: Literal["pending", "in_progress", "complete"]) -> str:
        """Update a task's status using its ID."""
        run_db(
            "UPDATE my_tasks SET status = :status WHERE id = :task_id",
            parameters={"status": status, "task_id": task_id},
        )
        rows = run_db(
            "SELECT id, title, status FROM my_tasks WHERE id = :task_id",
            parameters={"task_id": task_id},
            fetch=True,
        )
        if not rows:
            return f"No task found with ID {task_id}."
        row = rows[0]
        return f"Updated task {row['id']} ({row['title']}) to {row['status']}."

    @tool
    def delete_task(task_id: int) -> str:
        """Delete a task using its ID."""
        rows = run_db(
            "SELECT id, title FROM my_tasks WHERE id = :task_id",
            parameters={"task_id": task_id},
            fetch=True,
        )
        if not rows:
            return f"No task found with ID {task_id}."
        run_db(
            "DELETE FROM my_tasks WHERE id = :task_id",
            parameters={"task_id": task_id},
        )
        return f"Deleted task {task_id} ({rows[0]['title']})."

    return [add_task, list_tasks, update_task, delete_task]

st.subheader("AI Powered Task Manger, by Afraz 🤖")

if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()
    st.session_state.thread_id = str(uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# we have appended 2 things in "messages" 1. role, 2. content
# now we are getting it
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])


prompt = st.chat_input("I handle your tasks...")
if prompt:
    # user message part in browser
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role":"user", "content": prompt})

    agent = create_agent(
        model=model,
        tools=build_tools(),
        checkpointer=st.session_state.memory,
        system_prompt=system_prompt,
    )

    with st.chat_message("ai"):
        with st.spinner("Working on your request..."):
            result = agent.invoke(
                {"messages":[{"role": "user", "content":prompt}]},
                {"configurable": {"thread_id": st.session_state.thread_id}},
            )
        answer = result["messages"][-1].content
        if isinstance(answer, list):
            answer = "".join(
                block["text"]
                for block in answer
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            )
        st.markdown(answer)
        st.session_state.messages.append({"role":"ai", "content": answer})