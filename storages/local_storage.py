from llama_index.core import SimpleDirectoryReader

class LocalStorage:
    def __init__(self, directory):
        self.directory = directory
        self.loader = SimpleDirectoryReader(directory=directory)
    def get_loader(self):
        return self.loader