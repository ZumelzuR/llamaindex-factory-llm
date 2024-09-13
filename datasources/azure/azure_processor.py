# Implement the Azure class
import os
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore
from llama_index.vector_stores.azureaisearch import IndexManagement
from datasources.data_processor import DataProcessor
from llama_index.core import Settings, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
# create enum with HugginFace or AzureOpenAi

from engines.engine_ai import EngineAI
from services.ingestion_service import AzureStorage, IngestionService, LocalStorage, S3Storage
from domain.models import IndexItem


class AzureProcessor(DataProcessor):
    def __init__(self, builder: EngineAI, system_prompt=None):
        """
        Constructor for the Azure class.

        Parameters:
        - builder: An instance of the EngineAI class.
        - storage: An instance of either AzureStorage, S3Storage, or LocalStorage class. This is because the engine for vector dont have to be the same place than the storage
        - system_prompt: Optional. A string representing the system prompt for the agent.
        """
        load_dotenv()
        self.tools  = None
        self.local_indexes = []
        self.query_engine_tool = None
        self.system_prompt = system_prompt or """ \
                    You are an assistant that will have to answer question related to some knowledge base.
                    Please always use the tools provided to answer a question. Do not rely on prior knowledge.\
                    If some question you can't answer because need more details to decide where to took the answer from, ask to the user to be more specific\
                    """
        self.individual_query_engine_tools = []
        self.search_service_endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT")
        self.search_service_api_key = os.getenv("SEARCH_SERVICE_KEY")
        
        # init agent engine builder config
        self.builder = builder
        config = builder.get_config()

        Settings.llm = config['llm']
        Settings.embed_model  = config['embed_model']
        Settings.num_output = config['num_output']
        Settings.context_window = config['context_window']
        
        search_creds = AzureKeyCredential(self.search_service_api_key)
        self.index_client = SearchIndexClient(endpoint=self.search_service_endpoint, credential=search_creds)


    def init_store(self, indexes: list[IndexItem] = None):
        """
        Initializes the store with the specified indexes.
        Args:
            indexes (list, optional): A list of indexes (name and description) to initialize the store with. 
                If not provided, the method will retrieve the indexes from the index client.
        Returns:
            Void
        """
        self.local_indexes: list[IndexItem] = indexes
        if indexes is None or len(indexes) == 0:
            indexes = self.index_client.list_indexes()
            self.local_indexes = [IndexItem(i.name, '') for i in indexes]
        
        if not self.local_indexes or len(self.local_indexes) == 0:
            print("No indexes found")
            return
        
        for i in self.local_indexes:
            self.init_index(i.name, i.description)

    def setup_index(self, name):
        vector_store = AzureAISearchVectorStore(
            search_or_index_client=self.index_client,
            index_name=name,
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
    
    def setup_query_engine(self, name, description):
        index_vector = self.setup_index(name)
        self.individual_query_engine_tools.append(
            QueryEngineTool(
                query_engine=index_vector.as_query_engine(),
                metadata=ToolMetadata(
                    name=name,
                    description=description,
                )
            )
        )

        query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=self.individual_query_engine_tools,
        )

        self.query_engine_tool = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="sub_question_query_engine",
                description="useful for when you want to answer queries that require analyzing multiple documents from different contexts",
            ),
        )

        self.tools = self.individual_query_engine_tools + [self.query_engine_tool]
        return self.tools
    
    def setup_agent(self, tools):
        self.agent = OpenAIAgent.from_tools(tools, system_prompt=self.system_prompt, verbose=True)

    def init_index(self, index_name, description):
        self.setup_query_engine(index_name, description)
        self.setup_agent(self.tools)

    def ingest_data(self, index_name, storage: AzureStorage | S3Storage | LocalStorage):
        vector_store = AzureAISearchVectorStore(
            search_or_index_client=self.index_client,
            index_name=index_name,
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
        self.storage_service = IngestionService(loader=storage, embedding_model=Settings.embed_model, vector_store=vector_store)
        self.storage_service.run_ingestion()
        # TODO refresh indexers with setup indexes self.local_indexes
        return

    # def post_message(self, message):
    #     if not self.agent:
    #         raise Exception("Agent not setup")
    #     response = self.agent.chat(message)
    #     return str(response)
    
    def set_tools(self, tools):
        self.tools = tools
    
    def get_tools(self):
        return self.tools
    
    def get_system_prompt(self):
        return self.system_prompt
    
    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt

    def get_agent(self):
        return self.agent
    
    def set_agent(self, agent):
        self.agent = agent
        


