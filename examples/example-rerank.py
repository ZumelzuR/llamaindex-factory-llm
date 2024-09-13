# https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/chatbots/building_a_chatbot/

# usefull for mix results

import os
import openai
from llama_index.core import Settings, StorageContext
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI

import nest_asyncio

nest_asyncio.apply()

from llama_index.readers.file import UnstructuredReader
from pathlib import Path
filename_fn = lambda filename: {"file_name": filename}

Settings.llm = AzureOpenAI(
    engine="ai-model-nz-gpt4",
    model="gpt-4", temperature=0.01
)

Settings.embed_model  = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    engine="text-embedding-ada-002",
)

from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=256)

receipt_reader1 = SimpleDirectoryReader(
    input_dir="./data/segmentacion",
    recursive=True
).load_data()

index1 = VectorStoreIndex.from_documents(receipt_reader1, transformations=[splitter])

QUERY_GEN_PROMPT = (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Generate {num_queries} search queries, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
)

from llama_index.retrievers.bm25 import BM25Retriever

vector_retriever = index1.as_retriever(similarity_top_k=2)

bm25_retriever = BM25Retriever.from_defaults(
    docstore=index1.docstore, similarity_top_k=2
)
from llama_index.core.retrievers import QueryFusionRetriever

retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=2,
    num_queries=4,  # set this to 1 to disable query generation
    mode="reciprocal_rerank",
    use_async=True,
    verbose=True,
    # query_gen_prompt="...",  # we could override the query generation prompt here
)

from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(retriever)
response = query_engine.query("How I create an user?")
print(response)