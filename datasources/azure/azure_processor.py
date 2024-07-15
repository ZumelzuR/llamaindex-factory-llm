# Implement the Azure class
import os
from dotenv import load_dotenv
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore
from llama_index.vector_stores.azureaisearch import IndexManagement
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings,StorageContext
from datasources.data_processor import DataProcessor


class Azure(DataProcessor):
    def __init__(self):
        load_dotenv()
        self.search_service_endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT")
        self.search_service_api_key = os.getenv("SEARCH_SERVICE_KEY")


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
        
        search_creds = AzureKeyCredential(self.search_service_api_key)
        self.index_client = SearchIndexClient(endpoint=self.search_service_endpoint, credential=search_creds)

    def ingest_data(self, index, datapath):
        self.setup_index(index)
        docs = SimpleDirectoryReader(input_dir=datapath, recursive=True).load_data()
        VectorStoreIndex.from_documents(docs, storage_context=self.storage_context)
    
    def setup_index(self, index):
        vector_store = AzureAISearchVectorStore(
            search_or_index_client=self.index_client,
            index_name=index,
            index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
            id_field_key="id",
            chunk_field_key="chunk",
            embedding_field_key="embedding",
            embedding_dimensionality=1536,
            metadata_string_field_key="metadata",
            doc_id_field_key="doc_id",
            language_analyzer="en.lucene",
            vector_algorithm_type="exhaustiveKnn",
        )
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(vector_store, storage_context=self.storage_context)

        return self.index

