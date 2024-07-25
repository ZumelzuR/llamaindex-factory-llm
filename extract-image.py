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


image_parser =ImageReader(
    keep_image=True,
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
    input_dir="./data/example",
    file_metadata=filename_fn,
    file_extractor=file_extractor,
)

receipt_documents = receipt_reader.load_data()

print(len(receipt_documents))

print(receipt_documents[21])
