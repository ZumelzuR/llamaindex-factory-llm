from abc import ABC, abstractmethod

class EngineAI(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    def __init__(self, engine, model, temperature, embedding_model, embedding_engine):
        super.__init__(self, engine, model, temperature, embedding_model, embedding_engine)
        pass

    # ingest data into an index from a given datapath of documents
    @abstractmethod
    def get_config(self):
        pass