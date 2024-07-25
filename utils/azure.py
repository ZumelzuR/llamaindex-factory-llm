from azure.storage.blob import BlobServiceClient
from datasources.data_processor import DataProcessor, IndexItem
from services.ingestion_service import AzureStorage
import os
from dotenv import load_dotenv
load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.environ["AZURE_STORAGE_CONNECTION_STRING"]


def create_indexes_from_containers(data_processor: DataProcessor, containers: list = None, pattern: str = "-fake"):
    """
    Create indexes from a list of containers (or all) from an Azure Storage Account.

    Args:
        data_processor (DataProcessor): The data processor object.
        containers (list, optional): List of container names to create. If None, all containers will use to create indexes.
        pattern (str, optional): The pattern to append to container name for index name. Defaults to "-fake".
    """
    blob_service = BlobServiceClient.from_connection_string(conn_str=AZURE_STORAGE_CONNECTION_STRING)
    containers = blob_service.list_containers()
    index_list: list[IndexItem] = []
    for container in containers:
        index_list.append(IndexItem(container.name + pattern, "", container.name))

    if containers:
        index_list = [item for item in index_list if item.container in containers]

    data_processor.init_store(index_list)
    for index in index_list:
        storage = AzureStorage(index.container, AZURE_STORAGE_CONNECTION_STRING)
        data_processor.ingest_data(index.name, storage)
        print("Indexes created successfully")

# migrate_data_from_azure_to_azure(data_processor, pattern="-fake")
