from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from typing import List, Annotated

# Class for state -> graph state -> llm,memory -> node/function -> add node -> add edge
# -> compile graph -> invoke  


class ChatBotState(BaseModel):
    messages: Annotated[List, add_messages]

# create a state to store data
graphState = StateGraph(ChatBotState)

# llm
llm = ChatAnthropic(model="claude-opus-5")

# memory
memory = InMemorySaver()

# node
def chatNode(state:ChatBotState) -> ChatBotState:
    res = llm.invoke(state.messages)
    state.messages = [res]
    return state

# create a node
graphState.add_node("chatNode", chatNode)

# adding edge
graphState.add_edge(START, "chatNode")
graphState.add_edge("chatNode", END)

# compiling
graph = graphState.compile(checkpointer=memory)

# invoke
while True:
    query = input("User: ")
    if query.lower() in ["exit", "bye", "e"]:
        break

    config_memory = {"configurable":{"thread_id":1}}
    ans = graph.invoke(
            {
                "messages": [{"role": "user", "content": query}]
            },
            config_memory    
        )

    print("AI: ", ans["messages"][-1].content)