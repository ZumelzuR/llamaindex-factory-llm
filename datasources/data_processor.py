from abc import ABC, abstractmethod

class DataProcessor(ABC):
    @abstractmethod
    def __init__(self):
        pass

    # todo overload with index
    @abstractmethod
    def setup_index(self, index):
        pass

    # ingest data into an index from a given datapath of documents
    @abstractmethod
    def ingest_data(self, index, datapath):
        pass

    # get the query engine for a given index
    def get_query_engine(self, index):
        self.setup_index(index)
        index = self.index
        return index.as_query_engine()