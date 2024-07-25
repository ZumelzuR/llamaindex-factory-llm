

# Implement the Azure class
import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from engines.engine_ai import EngineAI

class StandarOpenAIBuilder(EngineAI):
    def __init__(self, model="gpt-4", temperature=0.01, embedding_engine="text-embedding-3-small"):
        load_dotenv()
        self.llm = OpenAI(
            model=model,
            temperature=temperature
        )

        self.embed_model  = OpenAIEmbedding(
            model=embedding_engine,
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
