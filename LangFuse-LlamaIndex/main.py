# I think the best way might be to use the LangFuse SDK to log the LLM calls
from llama_index.llms.ollama import Ollama

# For the Langfuse integration
from langfuse.decorators import observe, langfuse_context
import os
from langfuse import Langfuse

# Gotta add my credentials
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-5b1f3e97-6ab0-479a-9137-1fa51ef3e93e"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-be6ca1b4-4a43-4d19-8549-65db403379f5"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"

langfuse = Langfuse()

# Initialize llm
MODEL_NAME = "llama3:8b"
llm = Ollama(model=MODEL_NAME, request_timeout=120.0)


@observe(as_type="generation")
def llama_completion(
    input_topic: str,
) -> str:
    """
    Perform LlamaIndex completion for the given input topic.

    Args:
        input_topic (str): The topic for which completion is requested.

    Returns:
        str: The completion response from LlamaIndex.
    """
    prompt = langfuse.get_prompt("my-prompt")
    # Insert variables into prompt template
    compiled_prompt = prompt.compile(input="LangFuse")
    response = llm.complete(compiled_prompt)
    print(response)
    # Log the output
    langfuse_context.update_current_observation(
        prompt=prompt,  # Need to add in the prompt to link it to the observation
        model=MODEL_NAME,
        metadata={"task": "LlamaIndex completion"},
    )
    return str(response)  # It needs to be a string in order to be logged


@observe()
def main():
    response = llama_completion("LangFuse")
    print(response)


if __name__ == "__main__":
    main()
