# https://docs.llamaindex.ai/en/stable/examples/retrievers/reciprocal_rerank_fusion/
# https://learn.deeplearning.ai/courses/building-agentic-rag-with-llamaindex/lesson/5/building-a-multi-document-agent
# https://docs.llamaindex.ai/en/v0.10.17/examples/agent/openai_agent_retrieval.html
# usefull for isolated results

import os
import openai
from llama_index.core import Settings, StorageContext
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI

load_dotenv()


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

doc_set = {}
all_docs = []

receipt_reader1 = SimpleDirectoryReader(
    input_dir="./data/segmentacion/folder1",
)

receipt_reader2 = SimpleDirectoryReader(
    input_dir="./data/segmentacion/folder2",
)

docs1 = receipt_reader1.load_data(
)

docs2 = receipt_reader2.load_data(
)

Settings.chunk_size = 512
index_set = {}
# for year in years:

storage_context = StorageContext.from_defaults()
cur_index1 = VectorStoreIndex.from_documents(
    docs1,
    storage_context=storage_context,
)
storage_context.persist(persist_dir=f"./storage/folder1")


storage_context = StorageContext.from_defaults()
cur_index2 = VectorStoreIndex.from_documents(
    docs2,
    storage_context=storage_context,
)
storage_context.persist(persist_dir=f"./storage/folder2")



# storage_context = StorageContext.from_defaults(
#     persist_dir=f"./storage/{year}"
# )
# cur_index = load_index_from_storage(
#     storage_context,
# )
# index_set[year] = cur_index


index_set = [cur_index1, cur_index2]

from llama_index.core.tools import QueryEngineTool, ToolMetadata

individual_query_engine_tools = [
    QueryEngineTool(
        query_engine=index_set[0].as_query_engine(),
        metadata=ToolMetadata(
            name=f"vector_index_{0}",
            description=f"useful for when you want to answer queries about user interface manual instructions",
        ),
    ),
    QueryEngineTool(
        query_engine=index_set[1].as_query_engine(),
        metadata=ToolMetadata(
            name=f"vector_index_{1}",
            description=f"useful for when you want to answer queries about API calls",
        ),
    ),
]





query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=individual_query_engine_tools,
)

query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="sub_question_query_engine",
        description="useful for when you want to answer queries that require analyzing multiple documents from different contexts",
    ),
)

from llama_index.agent.openai import OpenAIAgent
tools = individual_query_engine_tools + [query_engine_tool]

agent = OpenAIAgent.from_tools(tools, system_prompt=""" \
                                You are an assistant that will have to answer question related to some knowledge base.
                                Please always use the tools provided to answer a question. Do not rely on prior knowledge.\
                                If some question you can't answer because need more datils to decide where to took the answer from, ask to the user to be more specific\
                                """,
                                verbose=True)

response = agent.chat("hi, how can I create an user?")
print(str(response))

# https://docs.llamaindex.ai/en/v0.10.17/examples/agent/openai_agent_retrieval.html
# todo check 