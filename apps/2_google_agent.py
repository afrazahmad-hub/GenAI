from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
# from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# create an object of "Google Serper"
search = GoogleSerperAPIWrapper()

# llm model
# model = ChatAnthropic(model="claude-opus-5")
model = ChatGroq(model="openai/gpt-oss-20b")
memory = MemorySaver()

agent = create_agent(
    model=model,
    tools=[search.run],
    checkpointer=memory,
    system_prompt="You are and agent to anser the questions by searching on Google."
)

while True:
    query = input("User: ")

    if query.lower() in ["quiet", "exit", "e"]:
        print("Good Bye...")
        break

    res = agent.invoke({
        "messages": [
            {"role": "user", "content": query}
        ]},
        {"configurable": {"thread_id": "afraz_313"}},
    )

    print("AI: ", res["messages"][-1].content)