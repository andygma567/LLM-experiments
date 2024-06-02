# Example from LlamaIndex
# https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_personality/
# Also please see their simple chatbot example

from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.ollama import Ollama

# initialize llm
llm = Ollama(model="llama3:8b", request_timeout=120.0)

chat_engine = SimpleChatEngine.from_defaults(llm=llm, verbose=False)

# Example of streaming taken from LlamaIndex
# https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_context/#streaming-support

while True:
    text_input = input("User: ")
    if text_input == "exit":
        break
    response = chat_engine.stream_chat(text_input)
    print("Chatbot: ", end="")
    for r in response.response_gen:
        print(r, end="", flush=True)
    print()
