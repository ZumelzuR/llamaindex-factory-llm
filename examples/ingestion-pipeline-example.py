from llama_index.core.ingestion import IngestionCache
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)
from llama_index.core.node_parser import SentenceSplitter
import os
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
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
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.extractors import (
QuestionsAnsweredExtractor,
TitleExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.llms.openai import OpenAI

from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore
from llama_index.readers.google import GoogleDriveReader

load_dotenv()

import nest_asyncio
nest_asyncio.apply()

ingest_cache = IngestionCache(
    cache=RedisCache.from_host_and_port(host="127.0.0.1", port=6379),
    collection="my_test_cache",
)

embed_model  = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    engine="text-embedding-ada-002",
)
Settings.embed_model = embed_model
Settings.llm = AzureOpenAI(
            engine="ai-model-nz-gpt4",
            model="gpt-4", temperature=0.01
)

index_name = "ingestion-index"
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
pc_res = pc.list_indexes()

print(pc_res.indexes)
if not any(index_name == idx['name'] for idx in pc_res.indexes):
    pc.create_index(name=index_name, dimension=1536, metric="dotproduct", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index(index_name)

vector_store = PineconeVectorStore(
    pinecone_index=index,
    add_sparse_vector=True,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)


from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

text_splitter = TokenTextSplitter(chunk_size=512)

import re
from llama_index.core.schema import TransformComponent

class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.text = re.sub(r"[^0-9A-Za-z ]", "", node.text)
        return nodes
    
from llama_index.core.ingestion import IngestionPipeline
MONGO_URI = os.environ["MONGO_URI"]

pipeline = IngestionPipeline(
    transformations=[
        text_splitter,
        embed_model,
        TitleExtractor(),
    ],
    docstore=MongoDocumentStore.from_uri(uri=MONGO_URI),
    vector_store=vector_store,
    cache=ingest_cache,
    docstore_strategy=DocstoreStrategy.UPSERTS,
)

from llama_index.core import SimpleDirectoryReader


import logging
import sys


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.readers.azstorage_blob import AzStorageBlobReader

# https://docs.llamaindex.ai/en/stable/examples/data_connectors/simple_directory_reader_remote_fs/
# we could use a s3 or another bucket and attach the simple file reader to this remote repository
loader = AzStorageBlobReader(
    container_name="temp",
    connection_string="DefaultEndpointsProtocol=https;AccountName=secutixstorage;AccountKey=NRjO8EBRV3cBAVxuZHClOOQ7anhTfnT3n1HYE2FY5DTp53OzHAo6BAfuYIGdCCbvfPegoE2Mk0t4+AStdM1D5w==;EndpointSuffix=core.windows.net",
)

documents = loader.load_data()
# index = VectorStoreIndex.from_documents(documents)

# documents = SimpleDirectoryReader("./data/example/").load_data()
nodes = pipeline.run(documents=documents)

# [DONE], create index pipeline for chunching and indexing periodically - done, just do a job in jenkins or cron
# TODO check type of data that we could index
# TODO create store class with all this for ingestion from folder, file or storage
# TODO check scrap connector
# TODO check discord connector
# TODO check delete document from index

# TODO test use case with a lot of documents (over chroma or pinecone)
# TODO check multiagent, like to check where (which index) go to ask


