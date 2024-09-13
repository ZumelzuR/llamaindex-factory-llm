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


image_parser =ImageReader(
    keep_image=False,
    parse_text=True
    )

filename_fn = lambda filename: {"file_name": filename}
file_extractor = {
    ".jpg": image_parser,
    ".png": image_parser,
    ".jpeg": image_parser,
}

# pdf_parser  =PDFReader(
#     )

# file_extractor_pdf = {
#     ".pdf": pdf_parser,
# }

receipt_reader = SimpleDirectoryReader(
    input_dir="./data/images",
    file_metadata=filename_fn,
    file_extractor=file_extractor,
)

receipt_documents = receipt_reader.load_data()

print(len(receipt_documents))

print(receipt_documents)

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

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
index = VectorStoreIndex.from_documents(receipt_documents)

chat_engine = index.as_chat_engine()
response = chat_engine.chat("Who is?")
print(response)