# Example used from LlamaIndex
# https://langfuse.com/docs/get-started#log-your-first-llm-call-to-langfuse
# https://langfuse.com/guides/cookbook/integration_llama-index
# I think that maybe only Indexes and Queries are supported in Langfuse?
# It definitely doesn't work with the simple chat engine when I tried it.
# Also using the llm directly doesn't work with the stream_complete example.
# Also doesn't work for the regular .complete method.

# I think the best way might be to use the LangFuse SDK to log the LLM calls
from llama_index.llms.ollama import Ollama

# For the Langfuse integration
from langfuse.decorators import observe, langfuse_context
import os

# Gotta add my credentials
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-5b1f3e97-6ab0-479a-9137-1fa51ef3e93e"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-be6ca1b4-4a43-4d19-8549-65db403379f5"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"


# initialize llm
MODEL_NAME = "llama3:8b"
llm = Ollama(model=MODEL_NAME, request_timeout=120.0)

# Example of streaming taken from LlamaIndex
# https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_context/#streaming-support

# Here's the docs for the update_current_observation method
# https://python.reference.langfuse.com/langfuse/decorators#LangfuseDecorator.update_current_observation


# This is a simple chat engine that can be used to interact with the LLM
@observe(as_type="generation")
def llama_completion(
    input_text: str,
) -> str:
    # I think the function input/output are automatically logged
    # https://langfuse.com/docs/sdk/python/decorators#capturing-of-inputoutput
    # From manual testing, this will automatically log the input
    # but it won't log the output... hence, I need to cast the output to a string
    # Attempting to auto lot the streamed output from .stream_complete()
    # seems to log the object instead of the string.

    response = llm.complete(input_text)
    # Log the output

    langfuse_context.update_current_observation(
        model=MODEL_NAME,
        metadata={"task": "LlamaIndex completion"},
    )
    return str(response)  # It needs to be a string in order to be logged


@observe()
def main():
    input_text = "What is the best ice cream flavor? Please give a concise answer."
    response = llama_completion(input_text)
    print(response)


main()
