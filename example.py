from datasources.data_processor import IndexItem
from datasources.data_processor_factory import DataProcessorFactory
from engines.engine_ai_factory import EngineAIFactory
from services.ingestion_service import AzureStorage
import os
from dotenv import load_dotenv

from utils.azure import create_indexes_from_containers
load_dotenv(override=True)

processor_type = "pinecone" #"azure" "chroma" "pinecone" # TODO TEST THE CHROMA Y PINECONE
AZURE_STORAGE_CONNECTION_STRING = os.environ["AZURE_STORAGE_CONNECTION_STRING"]

# each container is associate with an index, in this case temp is associated with test-index
storage1 = AzureStorage("temp", AZURE_STORAGE_CONNECTION_STRING)
storage2 = AzureStorage("temp2", AZURE_STORAGE_CONNECTION_STRING)

builder = EngineAIFactory.get_engine('azure')
data_processor = DataProcessorFactory.get_data_processor(processor_type, builder)

# upload file to the container
def upload_file(file_name, file_path, storage):
    with open(file_path, "rb") as data:
        storage.upload_file(file_name, data)
        print("File uploaded successfully")

def test_upload_files():
    upload_file("user_manual", "./data/segmentacion/folder1/data.txt", storage1)
    upload_file("user_api", "./data/segmentacion/folder2/data.txt", storage2)

# test_upload_files()


# index_list: list[IndexItem] = [
#     IndexItem("xxxxexample1", "useful for when you want to answer queries about user interface manual instructions"),
#     IndexItem("xxxxexample2","useful for when you want to answer queries about API calls")
# ]
data_processor.init_store()
for i in data_processor.local_indexes:
    data_processor.ingest_data(i.name, storage1)

# # create_indexes_from_containers(data_processor, pattern="-fake")
# # # will chose between all indexes to answer the question, based on the description of the indexes
query_engine = data_processor.post_message("How to create an user?")
print(query_engine)
