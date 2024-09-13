from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import (
    ImageReader,
)

class LocalStorage:
    def __init__(self, directory):
        self.directory = directory

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
            input_dir=directory,
            file_metadata=filename_fn,
            file_extractor=file_extractor
        )
    
    def get_loader(self):
        return self.loader