from dotenv import load_dotenv
load_dotenv()

# load - split - enbed - vector store - 
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain.agents import create_agent

import streamlit as st

st.subheader("Afraz's Personal Chat-bot ...")


# Declaring the variables for "Session state"
### data in st session
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages  = []

def process_docs(path):
    loader = PyPDFDirectoryLoader(path)
    pdf = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    splited_pdf = splitter.split_documents(documents=pdf)

    embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    vector_store = InMemoryVectorStore.from_documents(
        documents=splited_pdf,
        embedding=embedding
    )

    llm = ChatAnthropic(model="claude-opus-5")

    @tool
    def retreve_context(query:str):
        """Retrieve documents relevant to a query from the knowledge base."""

        docs = vector_store.similarity_search(query=query)

        context = ""
        for doc in docs:
            context += doc.page_content + "\n"
            # context = doc.page_content + "\n\n"

        return context

    prompt = "You are a helpful agent, you will find the answer of the questions from given document, using" \
    "retreve_context function. "

    memory = InMemorySaver()

    agent  = create_agent(
        model=llm,
        tools=[retreve_context],
        system_prompt=prompt,
        checkpointer=memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True


# UI for document upload
if not st.session_state.document_uploaded:

    # upload file(s)
    files_uploaded = st.file_uploader(
        label="Select a PDF file",
        type=["pdf"],
        accept_multiple_files=True
    )

    # Handling file(s)
    if files_uploaded:
        with st.spinner("Processing..."):
            path = "./docs_files/"
            for file in files_uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())

            process_docs(path)
            # To rerun the script and update the interface to show Chat_UI
            st.rerun()

# UI for Chat
if st.session_state.document_uploaded and st.session_state.agent:

    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)

    query = st.chat_input("Ask any question related to uploaded documents...")
    if query:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        res = st.session_state.agent.invoke(
            {"messages":[{"role": "user", "content": query}]},
            {"configurable":{"thread_id":1}}
        )

        answer = res["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"ai", "content":answer})
