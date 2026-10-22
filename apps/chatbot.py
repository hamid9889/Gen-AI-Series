from dotenv import load_dotenv
load_dotenv()
from langchain_community.utilities import GoogleAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver


model = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()

agent = create_agent(
    model = model,
    tools = [search.run],
    system_propmt=" you are a agent can search any question on google"
)

while True:
    query = input("user")
    if query.lower() == "quit":
        print(" good bye ")
        break
    
    res = agent.invoke(
        {"messge":[{"role":"user", "content":query}]},
        {"configurable": {"thread_id":"1"}})
    
    print(res["message"][-1].content)             
                





