# loading ENV variables
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

# -----------------------------------------------------------------------
# SIMPLE EXECUATION
# question = "The current Catholic Church Pope ?"
# res = llm.invoke(question)
# print(res.text)

# TERMINAL------- USING WHILE LOOP ---- input from user
# while True:
#     user_query = input("User: ")

#     # to exit chat
#     if user_query.lower() in ["quit", "exit", "bye"]:
#         print("Good Bye...👋🏽")
#         break

#     response = llm.invoke(user_query)
#     print(f"AI: {response.text}")
# -----------------------------------------------------------------------

# STREAMLIT
# Main page title and descriptions
st.title("Afraz's AI Q&N Bot...🤖")
st.markdown("A Google Gemini Powred Question/Answer Chat Bot.")

# To Preserve the browser memory---use session_state
if "messages" not in st.session_state:
    st.session_state.messages = [] 

# fetch "role" and "content" and send to st.chat_message
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content) 

# User input 
query = st.chat_input("Ask me anything...")

if query:
    # User input append to "messages"
    st.session_state.messages.append({"role": "user", "content": query})
    # user message portion
    st.chat_message("user").markdown(query)
    res = llm.invoke(query) # sent user query to LLM for response

    # Ai response portion display
    st.chat_message("ai").markdown(res.text)
    # Ai response appent to "messages"
    st.session_state.messages.append({"role": "ai", "content": res.text})