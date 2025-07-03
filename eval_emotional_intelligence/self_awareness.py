from collections import defaultdict
import math
import re
from typing import Any

from loguru import logger
from tqdm import tqdm

from eval_emotional_intelligence.base import Tool
from utils.utils_hf_models import HFModel
from utils.utils_llm import LLMMessage, OpenAIGenerator


class SelfAwarenessEvaluator(Tool):
    def __init__(self, reflective_model: str, emotion_model: str, sentiment_model: str):
        self.reflective_model = self.initialize_model(reflective_model)
        self.emotion_model = self.initialize_model(emotion_model)
        self.sentiment_model = self.initialize_model(sentiment_model)

    def invoke(self, turn: str, speaker: str, session_history: list[dict[str, Any]]) -> Any:
        is_reflective = self._is_reflective(turn, speaker, session_history, use_history=True)
        sentiment = self._classify_sentiment(turn)
        emotion = self._classify_emotion(turn)
        return {
            "is_reflective": is_reflective,
            "sentiment": sentiment,
            "emotion": emotion,
        }

    def compute_score(self, conversation: dict[str, Any]) -> dict[str, Any]:
        # Initialize dictionaries for reflective counts and diversity
        reflective_counts = defaultdict(int)
        total_counts = defaultdict(int)
        emotion_counts = defaultdict(lambda: defaultdict(int))
        sentiment_counts = defaultdict(lambda: defaultdict(int))
        # Process conversation data
        for session_key, session_data in tqdm(conversation.items()):
            if re.match(r"^session_\d+$", session_key):
                for turn in session_data:
                    speaker = turn["speaker"]

                    # Reflective counts
                    if turn.get("is_reflective", False):
                        reflective_counts[speaker] += 1
                    total_counts[speaker] += 1

                    # Count emotion and sentiment labels
                    if "emotion" in turn:
                        emotion_counts[speaker][turn["emotion"]] += 1
                    if "sentiment" in turn:
                        sentiment_counts[speaker][turn["sentiment"]] += 1

        # Calculate reflective frequencies
        reflective_frequencies = {
            speaker: reflective_counts[speaker] / total_counts[speaker] if total_counts[speaker] > 0 else 0
            for speaker in total_counts
        }

        # Calculate diversity (entropy) for emotions and sentiments
        def calculate_diversity(counts):
            diversities = {}
            for speaker, labels in counts.items():
                total = sum(labels.values())
                if total > 0:
                    probabilities = [count / total for count in labels.values()]
                    diversities[speaker] = -sum(p * math.log2(p) for p in probabilities if p > 0)
                else:
                    diversities[speaker] = 0.0
            return diversities

        emotion_diversities = calculate_diversity(emotion_counts)
        sentiment_diversities = calculate_diversity(sentiment_counts)

        # Return computed metrics
        return {
            "reflective_frequencies": reflective_frequencies,
            "emotion_diversities": emotion_diversities,
            "sentiment_diversities": sentiment_diversities,
        }

    def _is_reflective(self, text: str, speaker: str, session_history: list[dict[str, Any]], use_history: bool) -> bool:
        system_prompt = """You are an evaluator trained to determine if a speaker’s language is reflective, indicating self-awareness. Reflective language is characterized by self-observation, perspective-taking, and intentionality. This means that the speaker is not only aware of their thoughts, feelings, or actions but also able to express this awareness clearly.

A reflective response often includes one or more of the following traits:

Self-observation: The speaker describes their own emotional or cognitive state (e.g., “I feel uncertain about…” or “I’m aware that…”).
Perspective-taking: The speaker shows an understanding of how their actions or emotions affect others or acknowledges another person’s perspective on the situation (e.g., “I understand that my response may seem…”).
Intentionality: The speaker explains the reasoning behind their behavior or decisions, revealing their underlying motivations or goals (e.g., “I decided to respond this way because…”).

**Example Statements**

- "I realize I tend to get defensive when I receive feedback, and I think it’s because I want to do well."
Reflective or Not Reflective: Reflective
Reason: This statement shows self-observation (“I realize I tend to get defensive”) and insight into motivation (“because I want to do well”).

- "I did what I thought was best for the project."
Reflective or Not Reflective: Not Reflective
Reason: While the speaker describes their decision, they don’t analyze or acknowledge the emotions or motivations behind their choice or consider its impact on others.

"""

        if use_history:
            dialogue_context = "\n".join(
                f"{turn['speaker']}: {turn['clean_text'] if 'clean_text' in turn else turn['text']}"
                for turn in session_history
            )
            dialogue_prompt = f"""Given this dialogue context:
{dialogue_context}

Determine whether the {speaker}'s last message ('{text}') is reflective or not.
Reflective language includes 'I feel...', 'I think...', or similar reflective language.
Respond only with 'True' for reflective or 'False' for not reflective."""
        else:
            dialogue_prompt = f"""Determine whether the {speaker}'s last message ('{text}') is reflective or not.
Reflective language includes 'I feel...', 'I think...', or similar reflective language.
Respond only with 'True' for reflective or 'False' for not reflective."""

        reflective_detection_messages = [LLMMessage(role="system", content=system_prompt + dialogue_prompt)]
        try:
            assert isinstance(self.reflective_model, OpenAIGenerator)
            response = self.reflective_model.generate(messages=reflective_detection_messages)
            return "true" in response.content.lower().strip()
        except Exception as e:
            logger.error(f"[is_reflective] {e}")
            raise ValueError("Action (is_reflective) failed.") from e

    def _classify_emotion(self, text: str) -> Any:
        try:
            if isinstance(self.emotion_model, OpenAIGenerator):
                response = ""
            elif isinstance(self.emotion_model, HFModel):
                response = self.emotion_model.predict(text)["label"]
            else:
                raise ValueError("Invalid emotion model type.")
            return response
        except Exception as e:
            logger.error(f"[classify_emotion] {e}")
            return "neutral"

    def _classify_sentiment(self, text: str) -> Any:
        try:
            if isinstance(self.sentiment_model, OpenAIGenerator):
                response = ""
            elif isinstance(self.sentiment_model, HFModel):
                response = self.sentiment_model.predict(text)["label"]
            else:
                raise ValueError("Invalid sentiment model type.")
            return response
        except Exception as e:
            logger.error(f"[classify_sentiment] {e}")
            return "neutral"
