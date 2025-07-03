import re
from typing import Any

from loguru import logger
import numpy as np
from scipy.optimize import curve_fit

from eval_emotional_intelligence.base import Tool
from utils.utils_hf_models import HFModel
from utils.utils_llm import OpenAIGenerator


class SocialSkillsEvaluator(Tool):
    def __init__(self, intimacy_model: str):
        self.intimacy_model = self.initialize_model(intimacy_model)

    def invoke(self, turn: str, speaker: str, session_history: list[dict[str, Any]]) -> Any:
        intimacy = self._get_intimacy_score(turn)
        return {
            "intimacy": intimacy,
        }

    def compute_score(self, conversation: dict[str, Any]) -> dict[str, Any]:
        conversation_intimacy_scores = {}
        for session_key, session_data in conversation.items():
            session_intimacy_scores = {}
            if bool(re.match(r"^session_\d+$", session_key)):  # Filter session entries
                for _, turn in enumerate(session_data):
                    speaker = turn["speaker"]

                    # Initialize counts if the speaker is encountered for the first time
                    if speaker not in session_intimacy_scores:
                        session_intimacy_scores[speaker] = []

                    intimacy_score = turn["intimacy"]
                    session_intimacy_scores[speaker].append(intimacy_score)
                # Calculate the average intimacy score for each speaker
                for speaker, scores in session_intimacy_scores.items():
                    session_intimacy_scores[speaker] = sum(scores) / len(scores) if scores else 0.0

            for speaker, session_intimacy_score in session_intimacy_scores.items():
                if speaker not in conversation_intimacy_scores:
                    conversation_intimacy_scores[speaker] = []
                conversation_intimacy_scores[speaker].append(session_intimacy_score)

        intimacy_progression = {}
        for speaker, intimacy_scores in conversation_intimacy_scores.items():
            intimacy_progression[speaker] = self.compute_intimacy_progression(intimacy_scores)

        return intimacy_progression

    # Define model functions
    def _linear_model(self, x, a, b):
        return a * x + b

    def _exponential_model(self, x, a, b):
        return a * np.exp(b * x)

    def _polynomial_model(self, x, a, b, c):
        return a * x**2 + b * x + c

    def compute_intimacy_progression(self, session_intimacy_scores: list[float]):
        x_data = np.arange(1, len(session_intimacy_scores) + 1)

        # Fit linear model
        popt_linear, _ = curve_fit(self._linear_model, x_data, session_intimacy_scores)

        # Fit exponential model
        popt_exp, _ = curve_fit(self._exponential_model, x_data, session_intimacy_scores, maxfev=10000)

        # Fit polynomial model
        popt_poly, _ = curve_fit(self._polynomial_model, x_data, session_intimacy_scores)

        return {
            "linear_fit": {
                "linear_inclination": popt_linear[0],  # Slope
                # "initial_value": popt_linear[1],  # Intercept
            },
            "exponential_fit": {
                # "initial_value": popt_exp[0],  # a in a * e^(b*x)
                "growth_rate": popt_exp[1],  # b in a * e^(b*x)
            },
            "polynomial_fit": {
                "curvature": popt_poly[0],  # Coefficient for x^2
                "linear_coefficient": popt_poly[1],  # Coefficient for x
                # "initial_value": popt_poly[2],  # Intercept
            },
        }

    def _get_intimacy_score(self, text: str) -> Any:
        try:
            if isinstance(self.intimacy_model, OpenAIGenerator):
                response = ""
            elif isinstance(self.intimacy_model, HFModel):
                response = self.intimacy_model.predict(text)["score"]
            else:
                raise ValueError("Invalid intimacy model type.")
            return response
        except Exception as e:
            logger.error(f"[get_intimacy_score] {e}")
            return 0.0
