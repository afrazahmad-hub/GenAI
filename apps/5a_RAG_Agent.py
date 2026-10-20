from dotenv import load_dotenv
load_dotenv()

# load - split - enbed - vector store - 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import InMemoryVectorStore

from langchain.tools import tool
from langchain.agents import create_agent

loader = PyPDFLoader("../data/Afraz_ahmad_CV.pdf")
pdf = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
splited_pdf = splitter.split_documents(pdf)

embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector_store = InMemoryVectorStore.from_documents(
    documents=splited_pdf,
    embedding=embedding
)


@tool
def retreve_context(query:str):
    """Retrieve context from the vector store based on the query."""

    docs = vector_store.similarity_search(query=query)

    context = ""

    for doc in docs:
        context += doc.page_content + "\n"

    return context

prompt = "You are a helpful agent, you will find the answer of the questions from given document, using" \
"retreve_context function. "

llm = ChatAnthropic(model="claude-opus-5")

agent  = create_agent(
    model=llm,
    tools=[retreve_context],
    system_prompt=prompt
)


while True:
    query = input("User: ")

    if query.lower() == "exit":
        break


    res = agent.invoke({"messages": [{"role": "user", "content": query}]})
    ans = res["messages"][-1].content
    print("AI :",ans)