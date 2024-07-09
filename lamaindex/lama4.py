from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.replicate import Replicate
from transformers import AutoTokenizer


#Settings.llm = Ollama(model="llama2", request_timeout=300.0)

llm = Ollama(model="llama2", request_timeout=300.0)

service_context = ServiceContext.from_defaults(llm=llm)
pip install llama-index-embeddings-huggingface

# set tokenizer to match LLM
Settings.tokenizer = AutoTokenizer.from_pretrained(
    "NousResearch/Llama-2-7b-chat-hf"
)

# set the embed model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine()
response= query_engine.query("Was bedeuted Funktionelle Redundanz ?")


print(response)