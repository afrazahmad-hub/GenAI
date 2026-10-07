# Workflow
# LLm -> TOOLS: Google search Tools -> Agent -> Memory -> Streaming -> Web Interface (Streamlit)
# Streamlit Workflow
# Sub_header -> Chat_input -> Chat_message (Display "user" and "AI" messages in Browser) -> 
# session_state.memory (Presrve the chat memory) -> session_state.history (Presrve the chat history) ->
# for loop to save the "message" in the browser -> Streaming (chunks)

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

load_dotenv()

# llm model
model = ChatGroq(model="openai/gpt-oss-20b", streaming=True)

# tool
search = GoogleSerperAPIWrapper()
tools = [search.run]

# condition to check if "memory" does not exist in the session state memory
# if do not exist, then create one.
if "memory" not in st.session_state:
    # the problem with streamlit frontend is it dont store the history of conversation
    # each time we ask a question, it create a new object, which dont know the history of the conversation
    # to resolve this problem we use "session_state" method of "stream_lit"
    st.session_state.memory = MemorySaver()
    st.session_state.history = []

# create agent
agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt="You are an AI agent, will search on google to response."
)

st.subheader("Afraz's Chat Bot - Quick Answers 🚀")

# to display in browser
for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    # and save in chat_message (as it use to display the output in browswer)
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask anything ...")
if query:
    # to display "user" message in browser
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})

    # invoke agent and get answer
    response = agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": "7"}},
        stream_mode="messages"
    )

    # create a container in browser to store the response
    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty() # create an empty space for message write
        message = "" # Will start from blank space, then appended word/token by word/token.

        for chunk in response:
            message = message + chunk[0].content
            space.write(message)
        st.session_state.history.append({"role": "ai", "content": message})


    # answer = response["messages"][-1].content
    # # to display "AI" message in browser
    # st.chat_message("ai").markdown(answer)
    # st.session_state.history.append({"role": "ai", "content": answer})
