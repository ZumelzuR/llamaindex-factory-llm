from abc import ABC, abstractmethod

from engines.engine_ai import EngineAI
from services.ingestion_service import AzureStorage, LocalStorage, S3Storage
from domain.models import IndexItem

class DataProcessor(ABC):
    @abstractmethod
    def __init__(self, builder: EngineAI, system_prompt=None):
        pass

    @abstractmethod
    def init_store(self, indexes: list[IndexItem] = None):
        pass

    # ingest data into an index from a given datapath of documents
    @abstractmethod
    def ingest_data(self, index, storage: AzureStorage | S3Storage | LocalStorage):
        pass

    @abstractmethod
    def get_tools(self):
        pass
    
    @abstractmethod
    def get_system_prompt(self):
        pass

    @abstractmethod
    def set_system_prompt(self, system_prompt):
        pass

    @abstractmethod
    def get_agent(self):
        pass

    @abstractmethod
    def set_agent(self, agent):
        pass

    def post_message(self, message):
        if not self.agent:
            raise Exception("Agent not setup")
        response = self.agent.chat(message)
        return str(response)
