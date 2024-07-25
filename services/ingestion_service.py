import os
from llama_index.core.ingestion import IngestionCache
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.extractors import (
TitleExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import TransformComponent
from llama_index.core.ingestion import IngestionPipeline

from storages.azure_storage import AzureStorage
from storages.local_storage import LocalStorage
from storages.s3_storage import S3Storage
import re
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
load_dotenv()

import nest_asyncio
nest_asyncio.apply()

MONGO_URI = os.environ["MONGO_URI"]
REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PORT = os.environ["REDIS_PORT"]

class IngestionService:
    class TextCleaner(TransformComponent):
        def __call__(self, nodes, **kwargs):
            for node in nodes:
                node.text = re.sub(r"[^0-9A-Za-z ]", "", node.text)
            return nodes

    def __init__(self, loader: S3Storage | AzureStorage | LocalStorage, embedding_model: HuggingFaceEmbedding | AzureOpenAIEmbedding | OpenAIEmbedding, vector_store: PineconeVectorStore | AzureAISearchVectorStore | ChromaVectorStore):
        self.embed_model = embedding_model
        self.vector_store = vector_store
        self.loader = loader
        
        self.ingest_cache = IngestionCache(
            cache=RedisCache.from_host_and_port(host=REDIS_HOST, port=REDIS_PORT),
            collection="ingestion_cache",
        )
        self.text_splitter = TokenTextSplitter(chunk_size=512)


        self.pipeline = IngestionPipeline(
            transformations=[
                self.text_splitter,
                self.embed_model,
                TitleExtractor(),
            ],
            docstore=MongoDocumentStore.from_uri(uri=MONGO_URI),
            vector_store=self.vector_store,
            cache=self.ingest_cache,
            docstore_strategy=DocstoreStrategy.UPSERTS,
            disable_cache=True
        )

    def run_ingestion(self):
        documents = self.loader.get_loader().load_data()
        nodes = self.pipeline.run(documents=documents, show_progress=True)
        return nodes
