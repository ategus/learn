import os
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# Set up the Ollama LLM
Settings.llm = Ollama(model="llama2")

# Load documents from the "data" directory
documents = SimpleDirectoryReader('data').load_data()

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Create an index from the documents
index = VectorStoreIndex.from_documents(documents)

# Create a query engine
query_engine = index.as_query_engine()

# Function to handle user queries
def ask_question(question):
    response = query_engine.query(question)
    return response.response

# Main loop
if __name__ == "__main__":
    query_engine = index.as_query_engine()
    response= query_engine.query('Was bedeuted Funktionelle Redundanz ?')


    print(response)