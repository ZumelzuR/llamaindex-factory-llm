from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import Settings,StorageContext
from llama_index.llms.azure_openai import AzureOpenAI

from datasources.azure.azure_processor import Azure
from datasources.chroma.chroma_processor import Chroma
from datasources.pinecone.pinecone_processor import PineconeProcessor

# enum for processor types
class StoreType:
    AZURE = "azure"
    CHROMA = "chroma"
    PINECONE = "pinecone"

# Factory class to get the data processor
class DataProcessorFactory:

    @staticmethod
    def get_data_processor(processor_type):
        # embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        if processor_type == StoreType.AZURE:
            return Azure()
        elif processor_type == StoreType.CHROMA:
            return Chroma()
        elif processor_type == StoreType.PINECONE:
            return PineconeProcessor()
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")