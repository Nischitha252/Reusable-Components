from azure.storage.blob import BlobServiceClient

AZURE_STORAGE_CONNECTION_STRING = "your_connection_string"

class BlobStorageManager:
    def __init__(self):
        self.blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

    def upload_blob(self, container_name, blob_name, data):
        container_client = self.blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(data, overwrite=True)

    def download_blob(self, container_name, blob_name):
        container_client = self.blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)
        return blob_client.download_blob().readall()

def main():
    container_name = "your_container"
    blob_name = "your_file_name"
    file_to_upload = f"file_path_to_upload"
    file_to_download = f"file_path_to_download"

    manager = BlobStorageManager()
    
    # Upload blob
    with open(file_to_upload, "rb") as data:
        manager.upload_blob(container_name, blob_name, data)
    print(f"Uploaded blob '{blob_name}' to container '{container_name}'.")

    # Download blob
    downloaded_data = manager.download_blob(container_name, blob_name)
    with open(file_to_download, "wb") as file:
        file.write(downloaded_data)
    print(f"Downloaded blob '{blob_name}' from container '{container_name}' and saved to '{file_to_download}'.")

if __name__ == "__main__":
    main()