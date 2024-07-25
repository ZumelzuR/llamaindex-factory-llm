from llama_index.readers.azstorage_blob import AzStorageBlobReader
from azure.storage.blob import BlobServiceClient

class AzureStorage:
    def __init__(self, container_name, connection_string):
        self.container_name = container_name
        self.container_connection_string = connection_string
        self.loader = AzStorageBlobReader(
                container_name=container_name,
                connection_string=connection_string,
        )
        self.blob_service_client = BlobServiceClient.from_connection_string(conn_str=connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        if not self.container_client.exists():
            self.container_client = self.blob_service_client.create_container(container_name)

    def upload_file(self, file_name, file_content):
        self.container_client.upload_blob(name=file_name, data=file_content)

    def get_loader(self):
        return self.loader
    
    def get_storage(self):
        return self.container_client