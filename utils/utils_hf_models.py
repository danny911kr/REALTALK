from typing import Any

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.pipelines import pipeline


HF_AVAILABLE_MODELS = {
    "hf": [
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "cardiffnlp/twitter-roberta-large-intimacy-latest",
        "cardiffnlp/twitter-roberta-large-emotion-latest",
        "cardiffnlp/twitter-roberta-base-2021-124m-sentiment",
    ],
}


class HFModel:
    def __init__(self, model_name: str):
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

    def predict(self, text: str) -> dict[str, Any]:
        predictions = self.model_pipeline(text)

        # Runtime check and safe access
        if not isinstance(predictions, list) or not predictions or not isinstance(predictions[0], dict):
            raise ValueError("Unexpected prediction output format from pipeline.")

        return predictions[0]
