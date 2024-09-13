from llama_index.readers.azstorage_blob import AzStorageBlobReader
from azure.storage.blob import BlobServiceClient
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from s3fs import S3FileSystem
import boto3
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

class S3Storage:
    def __init__(self, bucket, key, aws_access_id, aws_access_secret):
        self.bucket = bucket
        self.key = key
        self.aws_access_id = aws_access_id
        self.aws_access_secret = aws_access_secret

        endpoint_url = None
        s3 = boto3.resource("s3", endpoint_url=endpoint_url)
        # s3.create_bucket(Bucket=self.bucket)
        bucket = s3.Bucket(self.bucket)
        s3_fs = S3FileSystem(anon=False, endpoint_url=endpoint_url)

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

        self.loader = SimpleDirectoryReader(
            input_dir=self.bucket,
            fs=s3_fs,
            file_metadata=filename_fn,
            file_extractor=file_extractor,
            recursive=True,  # recursively searches all subdirectories
        )

    def get_loader(self):
        return self.loader