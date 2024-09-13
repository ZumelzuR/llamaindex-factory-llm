from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)
from llama_index.core.node_parser import SentenceSplitter
import os
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
from llama_index.embeddings.openai import OpenAIEmbedding
import fitz

# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# Settings.llm = AzureOpenAI(
#     engine="ai-model-nz-gpt4",
#     model="gpt-4", temperature=0.01
# )

embed_model  = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    engine="text-embedding-ada-002",
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
    pinecone_index=index_name,
    add_sparse_vector=True,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

if vector_store.index_exists():
    vector_store.delete_index()

text_parser = SentenceSplitter(
    chunk_size=1024,
    # separator=" ",
)

def split_document(doc):
    text_chunks = []
    doc_idxs = []
    for doc_idx, page in enumerate(doc):
        page_text = page.get_text("text")
        cur_text_chunks = text_parser.split_text(page_text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc_idx = doc_idxs[idx]
        src_page = doc[src_doc_idx]
        nodes.append(node)
    return nodes



llm = AzureOpenAI(
    engine="ai-model-nz-gpt4",
    model="gpt-4", temperature=0.01
)

extractors = [
    TitleExtractor(nodes=5, llm=llm),
    # QuestionsAnsweredExtractor(questions=3, llm=llm),
    # OpenAIEmbedding(),
]

pipeline = IngestionPipeline(
    transformations=extractors,
    # vector_store=vector_store,
)

file_path = "./data/attention_is_all_you-need.pdf"
doc = fitz.open(file_path)

nodes = split_document(doc)

nodes = pipeline.arun(nodes=nodes, in_place=False)

for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding
vector_store.add(nodes)
