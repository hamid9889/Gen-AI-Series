from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver

# --- Database setup ---
db = SQLDatabase.from_uri("sqlite:///my_task.db")

db.run("""
    CREATE TABLE IF NOT EXISTS task(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL, 
        description TEXT, 
        status TEXT CHECK(status IN ('pending', 'in_progress', 'complete')) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")

print("DB CREATED")

# --- Model and Toolkit setup ---
model = ChatGroq(model="openai/gpt-oss-20b")
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
memory = InMemorySaver()

system_prompt = """
You are a task management assistant that interacts with a SQL database containing a 'tasks' table.

TASK RULES:
1. Limit SELECT queries to 10 results max with ORDER BY created_at DESC
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If the user requests a list of tasks, present the output in a structured table format

CRUD OPERATIONS:
CREATE: INSERT INTO tasks(title, description, status)
READ: SELECT * FROM tasks WHERE ... LIMIT 10
UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
DELETE: DELETE FROM tasks WHERE id=? OR title=?
"""

agent = create_agent(
    model=model, 
    tools=tools, 
    checkpointer=memory, 
    system_prompt=system_prompt
)

# --- Interactive loop ---
while True:
    query = input("user: ")
    response = agent.invoke({
        "messages": [{"role": "user", "content": query}],
        "configurable": {
            "thread_id": "thread_1",        # unique conversation ID
            "checkpoint_ns": "sql_agent",   # namespace for checkpoints
            "checkpoint_id": "session_1"    # unique session ID
        }
    })
    result = response["messages"][-1].content
    print("Ai:", result)

# --- Print available tools ---
for tool in tools:
    print(tool.name)
  