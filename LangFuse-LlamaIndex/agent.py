from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec

tool_spec = DuckDuckGoSearchToolSpec()

# initialize llm
llm = Ollama(model="llama3:8b", request_timeout=120.0)

# initialize ReAct agent
agent = ReActAgent.from_tools(
    tool_spec.to_tool_list(), llm=llm, verbose=False
)

while True:
    text_input = input("User: ")
    if text_input == "exit":
        break
    response = agent.chat(text_input)
    print(f"Agent: {response}")
