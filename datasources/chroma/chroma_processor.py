# Implement the Azure class
import os
import chromadb
from dotenv import load_dotenv
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore
from llama_index.vector_stores.azureaisearch import IndexManagement
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings,StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from datasources.data_processor import DataProcessor

class Chroma(DataProcessor):
    def __init__(self):
        self.db = chromadb.PersistentClient(path="./storage/chroma")

         ## TODO setup the LLM and Embedding model in constructor to use azure or another
        Settings.llm = AzureOpenAI(
            engine="ai-model-nz-gpt4",
            model="gpt-4", temperature=0.01
        )

        Settings.embed_model  = AzureOpenAIEmbedding(
            model="text-embedding-ada-002",
            engine="text-embedding-ada-002",
        )

        Settings.num_output = 512
        Settings.context_window = 3900

    def ingest_data(self, index, datapath):
        self.setup_index(index)
        docs = SimpleDirectoryReader(input_dir=datapath).load_data()
        VectorStoreIndex.from_documents(docs, storage_context=self.storage_context, show_progress=True)
        

    def setup_index(self, index):
        chroma_collection = self.db.get_or_create_collection(index)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(vector_store, storage_context=self.storage_context)
            
        return self.index
