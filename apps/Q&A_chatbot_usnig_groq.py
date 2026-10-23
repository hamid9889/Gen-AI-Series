""" 
llm 
tool google swearch 
memory 
streaming
web iterface 

"""

from dotenv import load_dotenv
load_dotenv()
from langchain_community.utilities import GoogleAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

llm =  ChatGroq(model= "openai/gpt-oss-20b" , streaming = True)
search = GoogleAPIWrapper()
tools = [search.run]



if " memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.memory = []



agent = create_agent(
    model = llm, 
    tools = tools,
    checkpointer = st.session_state.memory, 
    system_propmt = " you are a amazing ai search on google "
)

# building the agent code
st.subheader(" Quick answer = here the output ")


for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)
    
    
query = st.chat_input(" Ask anything")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.memory.append({"role":"user", "content":query})





res = agent.invoke(
        {"messge":[{"role":"user", "content":query}]},
        {"configurable": {"thread_id":"1"}})
    
ai_container = st.chat_message("ai")
with ai_container:
    space = st.empty()
    
    message = ""
    for chunk in res:
        message = message+chunk[0].content
        space.write(message)
    
    
    st.session_state.history.append({"role":"ai", "content":message})
            



