

from llama_index.readers.nougat_ocr import PDFNougatOCR
from llama_index.core import SimpleDirectoryReader
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'./venv/bin/pytesseract'

text = pytesseract.image_to_string(Image.open('image.png'))
print(text)
# reader = PDFNougatOCR()
# file_extractor_pdf = {
#     ".pdf": reader,
# }
# receipt_reader = SimpleDirectoryReader(
#     input_dir="./data/example",
#     file_extractor=file_extractor_pdf,
# )
# receipt_documents = receipt_reader.load_data()
# print(len(receipt_documents))
# print(receipt_documents[0])

# pdf_path = Path("/path/to/pdf")

# documents = reader.load_data(pdf_path)

# image_parser =ImageReader(
#     keep_image=True,
#     parse_text=True
#     )

# filename_fn = lambda filename: {"file_name": filename}
# file_extractor = {
#     ".jpg": image_parser,
#     ".png": image_parser,
#     ".jpeg": image_parser,
# }

# # pdf_parser  =PDFReader(
# #     )

# # file_extractor_pdf = {
# #     ".pdf": pdf_parser,
# # }

# receipt_reader = SimpleDirectoryReader(
#     input_dir="./data/example",
#     file_metadata=filename_fn,
#     file_extractor=file_extractor,
# )

# receipt_documents = receipt_reader.load_data()

# print(len(receipt_documents))

# print(receipt_documents[21])
