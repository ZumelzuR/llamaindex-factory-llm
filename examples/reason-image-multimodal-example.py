from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal
import base64
import requests
from llama_index.core.schema import ImageDocument

image_url = "https://www.visualcapitalist.com/wp-content/uploads/2023/10/US_Mortgage_Rate_Surge-Sept-11-1.jpg"
headers ={
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
}
response = requests.get(image_url, headers=headers)
if response.status_code != 200:
    raise ValueError("Error: Could not retrieve image from URL.")
base64str = base64.b64encode(response.content).decode("utf-8")

image_document = ImageDocument(image=base64str, image_mimetype="image/jpeg")

azure_openai_mm_llm = AzureOpenAIMultiModal(
    engine="gpt-4-vision-preview",
    api_version="2023-12-01-preview",
    model="gpt-4-vision-preview",
    max_new_tokens=300,
)

complete_response = azure_openai_mm_llm.complete(
    prompt="Describe the images as an alternative text",
    image_documents=[image_document],
)

# in addition we could use this with  multimodal vector store in order to get the vectors of the image and the text
# and store this image index in different collection
# then when you ask something to the bot the bot will be able to give you as well an image as response if there are coincidences

# https://docs.llamaindex.ai/en/stable/examples/multi_modal/multi_modal_rag_nomic/

print(complete_response)