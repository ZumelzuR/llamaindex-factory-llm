# Implement the Azure class
import os
import chromadb
import pinecone
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
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from datasources.data_processor import DataProcessor

class PineconeProcessor(DataProcessor):
    def __init__(self):
        load_dotenv()
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=self.pinecone_api_key)

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
        documents = SimpleDirectoryReader(datapath).load_data()
        VectorStoreIndex.from_documents(
            documents, storage_context=self.storage_context
        )
    
    def setup_index(self, index):
        pc_res = self.pc.list_indexes()
        if not any(index == idx['name'] for idx in pc_res.indexes):
            self.pc.create_index(name=index, dimension=1536, metric="dotproduct", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        index = self.pc.Index(index)
        vector_store = PineconeVectorStore(
            pinecone_index=index,
            add_sparse_vector=True,
        )
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(vector_store, storage_context=self.storage_context)
        return self.index, self.storage_context
    