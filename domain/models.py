
class VectorStoreType():
    AZURE = "azure"
    CHROMA = "chroma"
    PINECONE = "pinecone"

class EngineType():
    AZURE = "azure"
    STANDARD = "standard"
    HUGGINGFACES = "HUGGINGFACES"

class IndexItem:
    def __init__(self, name: str, description: str, container: str = None):
        self.name = name
        self.description = description
        self.container = container


