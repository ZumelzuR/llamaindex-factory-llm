# Implement the Azure class
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from dotenv import load_dotenv
from llama_index.llms.azure_openai import AzureOpenAI

from engines.engine_ai import EngineAI

class AzureOpenAIBuilder(EngineAI):
    def __init__(self, engine="ai-model-nz-gpt4", model="gpt-4", temperature=0.01, embedding_model="text-embedding-ada-002", embedding_engine="text-embedding-ada-002"):
        load_dotenv()
        self.llm = AzureOpenAI(
            engine=engine,
            model=model,
            temperature=temperature
        )

        self.embed_model  = AzureOpenAIEmbedding(
            model=embedding_model,
            engine=embedding_engine,
        )

        self.num_output = 512
        self.context_window = 3900
        
    def get_config(self):
        return {
            "llm": self.llm,
            "embed_model": self.embed_model,
            "num_output": self.num_output,
            "context_window": self.context_window
        }