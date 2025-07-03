import re
from typing import Any

from loguru import logger
from tqdm import tqdm

from eval_emotional_intelligence.base import Tool
from utils.utils_llm import LLMMessage, OpenAIGenerator


class EmpathyEvaluator(Tool):
    def __init__(self, empathy_model: str):
        self.empathy_model = self.initialize_model(empathy_model)

    def invoke(self, turn: str, speaker: str, session_history: list[dict[str, Any]]) -> Any:
        empathy_score = self.get_empathy_score(turn, speaker, session_history, use_history=True)
        return {
            "empathy": empathy_score,
        }

    def compute_score(self, conversation: dict[str, Any]) -> dict[str, Any]:
        speaker_empathy_scores = {}

        # Iterate through sessions and turns
        for session_key, session_data in tqdm(conversation.items()):
            if bool(re.match(r"^session_\d+$", session_key)):  # Filter session entries
                for _, turn in enumerate(session_data):
                    speaker = turn["speaker"]
                    # Initialize counts if the speaker is encountered for the first time
                    if speaker not in speaker_empathy_scores:
                        speaker_empathy_scores[speaker] = []

                    empathy_score = turn["empathy"]
                    total_empathy_score = sum(empathy_score.values())
                    speaker_empathy_scores[speaker].append(total_empathy_score)

        # Compute average empathy score for each speaker
        empathy_scores = {}
        for speaker, scores in speaker_empathy_scores.items():
            empathy_scores[speaker] = sum(scores) / len(scores)

        return {
            "empathy_average": empathy_scores,
        }

    def get_empathy_score(
        self, text: str, speaker: str, session_history: list[dict[str, Any]], use_history: bool
    ) -> dict[str, int]:
        system_prompt = """You are an evaluator assessing the level of empathy conveyed in a response, based on three core components: Emotional Reaction, Interpretation, and Exploration. For each component, provide a score from 0–2, where 0 indicates no presence, 1 indicates partial presence, and 2 indicates explicit presence. Sum the scores from each component to obtain an overall empathy score.

**Component 1: Emotional Reaction**
Does the response express or allude to warmth, compassion, concern or similar feelings of the responder towards the seeker?
- 0: No.
- 1: Yes, the response alludes to these feelings but the feelings are not explicitly expressed.
- 2: Yes, the response has an explicit mention.

**Component 2: Interpretation**
Does the response communicate an understanding of the seeker’s experiences and feelings? In what manner?
- 0: No.
- 1: Yes, the response communicates an understanding of the seeker’s experiences and/or feelings.
    - The response contains conjectures or speculations about the seeker’s experiences and/or feelings.
    - The responder reflects back on similar experiences of their own or others.
    - The responder describes similar experiences of their own or others.
    - The response paraphrases the seeker’s experiences and/or feelings.
- 2: The response provides a deep, explicit understanding and validation of the seeker’s feelings or experiences, potentially using multiple sub-categories.

**Component 3: Exploration**
Does the response make an attempt to explore the seeker’s experiences and feelings?
- 0: No.
- 1: Yes, the exploration is present but remains generic.
- 2: Yes, the exploration is present and is specific, delving into the seeker’s particular feelings or experiences.

Return output in a following JSON format:
{{
    "emotional_reaction": [0–2],
    "interpretation": [0–2],
    "exploration": [0–2],
}}
"""

        if use_history:
            dialogue_context = "\n".join(
                f"{turn['speaker']}: {turn['clean_text'] if 'clean_text' in turn else turn['text']}"
                for turn in session_history
            )
            dialogue_prompt = f"""Given this dialogue context:
{dialogue_context}

Score the empathy of {speaker}'s last message ('{text}') in JSON format."""
        else:
            dialogue_prompt = f"""Score the empathy of {speaker}'s last message ('{text}') in JSON format."""

        empathy_detection_messages = [LLMMessage(role="system", content=system_prompt + dialogue_prompt)]
        try:
            assert isinstance(self.empathy_model, OpenAIGenerator)
            response = self.empathy_model.generate_json(messages=empathy_detection_messages)
            return response.content
        except Exception as e:
            logger.error(f"[empathy] {e}")
            raise ValueError("Action (empathy) failed.") from e
