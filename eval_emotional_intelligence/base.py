import os
from typing import Any

from utils.utils_hf_models import HF_AVAILABLE_MODELS, HFModel
from utils.utils_llm import OpenAIGenerator


class Tool:
    def initialize_model(self, model: str):
        for _, models in HF_AVAILABLE_MODELS.items():
            if model in models:
                return HFModel(model)

        return OpenAIGenerator(
            model=model,
            oai_api_key=os.getenv("OPENAI_API_KEY", ""),
        )

    def invoke(self, turn: str, speaker: str, session_history: list[dict[str, Any]]) -> Any:
        raise NotImplementedError("Subclasses must implement this method.")

    def compute_score(self, conversation):
        raise NotImplementedError("Subclasses must implement this method.")
