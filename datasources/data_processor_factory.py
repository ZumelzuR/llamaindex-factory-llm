from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from datasources.azure.azure_processor import AzureProcessor
from datasources.chroma.chroma_processor import ChromaProcessor
from datasources.pinecone.pinecone_processor import PineconeProcessor
from domain.models import VectorStoreType

# Factory class to get the data processor
class DataProcessorFactory:

    @staticmethod
    def get_data_processor(processor_type, builder, system_prompt=None):
        if processor_type == VectorStoreType.AZURE:
            return AzureProcessor(builder, system_prompt)
        elif processor_type == VectorStoreType.CHROMA:
            return ChromaProcessor(builder, system_prompt)
        elif processor_type == VectorStoreType.PINECONE:
            return PineconeProcessor(builder, system_prompt)
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")