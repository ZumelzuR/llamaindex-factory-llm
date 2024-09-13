from llama_index.readers.file import (
    DocxReader,
    HWPReader,
    PDFReader,
    EpubReader,
    FlatReader,
    HTMLTagReader,
    ImageCaptionReader,
    ImageReader,
    ImageVisionLLMReader,
    IPYNBReader,
    MarkdownReader,
    MboxReader,
    PptxReader,
    PandasCSVReader,
    VideoAudioReader,
    UnstructuredReader,
    PyMuPDFReader,
    ImageTabularChartReader,
    XMLReader,
    PagedCSVReader,
    CSVReader,
    RTFReader,
)
from llama_index.core import SimpleDirectoryReader
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings, StorageContext
from dotenv import load_dotenv
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import SummaryIndex
from llama_index.readers.web import SimpleWebPageReader
from IPython.display import Markdown, display
import os


documents = SimpleWebPageReader(html_to_text=True).load_data(
    ["http://facebook.com"]
)
print((documents[0]))

def init( engine="ai-model-nz-gpt4", model="gpt-4", temperature=0.01, embedding_model="text-embedding-ada-002", embedding_engine="text-embedding-ada-002"):
    load_dotenv()
    Settings.llm = AzureOpenAI(
        engine=engine,
        model=model,
        temperature=temperature
    )

    Settings.embed_model  = AzureOpenAIEmbedding(
        model=embedding_model,
        engine=embedding_engine,
    )

    Settings.num_output = 512
    Settings.context_window = 3900

init()

print("Summary index")
index = SummaryIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is this?")
print(response)
