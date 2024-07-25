from domain.models import EngineType
from engines.builders.azure_openai_builder import AzureOpenAIBuilder
from engines.builders.hugging_openai_builder import HuggingOpenAIBuilder
from engines.builders.standar_openai_builder import StandarOpenAIBuilder

# Factory class to get the data processor
class EngineAIFactory:

    @staticmethod
    def get_engine(engine_type):
        if engine_type == EngineType.AZURE:
            return AzureOpenAIBuilder()
        elif engine_type == EngineType.STANDARD:
            return StandarOpenAIBuilder()
        elif engine_type == EngineType.HUGGINGFACES:
            return HuggingOpenAIBuilder()
        else:
            raise ValueError(f"Unknown processor type: {engine_type}")