from llama_index.llms.ollama import Ollama

llm = Ollama(model="llama2", request_timeout=300.0)

response = llm.complete("What is the capital of France?")
print(response)