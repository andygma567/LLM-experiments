from langfuse import Langfuse
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.ollama import Ollama

# Initialize Langfuse client
# I had to create an API key for this example
langfuse = Langfuse(
    secret_key="sk-lf-5b1f3e97-6ab0-479a-9137-1fa51ef3e93e",
    public_key="pk-lf-be6ca1b4-4a43-4d19-8549-65db403379f5",
    host="http://localhost:3000",
)

# Get current production version of a text prompt
# You can only fetch prompts labeled as "production"
# In the UI this means you need to activate the prompt
prompt = langfuse.get_prompt("my-prompt")

# Print prompt template
print(prompt.prompt)
print(type(prompt))  # <class 'langfuse.model.TextPromptClient'>
print()

# Insert variables into prompt template
compiled_prompt = prompt.compile(input="LangFuse")
print(compiled_prompt)
print(type(compiled_prompt))  # <class 'str'>

# Test the compiled prompt with the chat engine

# initialize llm
llm = Ollama(model="llama3:8b", request_timeout=120.0)

chat_engine = SimpleChatEngine.from_defaults(llm=llm, verbose=False)

response = chat_engine.stream_chat(compiled_prompt)
print("Chatbot: ", end="")
for r in response.response_gen:
    print(r, end="", flush=True)

# This appears to work and now I need to check that I can integrate this with the
# tracing feature of Langfuse.
