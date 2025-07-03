import re
from typing import Any

from loguru import logger
from tqdm import tqdm

from eval_emotional_intelligence.base import Tool
from utils.utils_llm import LLMMessage, OpenAIGenerator


class MotivationEvaluator(Tool):
    def __init__(self, grounding_model: str):
        self.grounding_model = self.initialize_model(grounding_model)

    def invoke(self, turn: str, speaker: str, session_history: list[dict[str, Any]]) -> Any:
        is_grounding = self._is_grounding(turn, speaker, session_history, use_history=True)
        return {
            "is_grounding": is_grounding,
        }

    def compute_score(self, conversation: dict[str, Any]) -> dict[str, Any]:
        speaker_grounding_counts = {}
        speaker_total_counts = {}

        # Iterate through sessions and turns
        for session_key, session_data in tqdm(conversation.items()):
            if bool(re.match(r"^session_\d+$", session_key)):  # Filter session entries
                for _, turn in enumerate(session_data):
                    speaker = turn["speaker"]

                    # Initialize counts if the speaker is encountered for the first time
                    if speaker not in speaker_grounding_counts:
                        speaker_grounding_counts[speaker] = 0
                        speaker_total_counts[speaker] = 0

                    # Count total turns for the speaker
                    speaker_total_counts[speaker] += 1

                    # Check if the turn is reflective
                    is_grounding = turn["is_grounding"]
                    if is_grounding:
                        speaker_grounding_counts[speaker] += 1

        grounding_frequencies = {}
        for speaker, grounding_count in speaker_grounding_counts.items():
            total_turns = speaker_total_counts[speaker]
            grounding_frequencies[speaker] = grounding_count / total_turns if total_turns > 0 else 0

        return {
            "grounding_frequencies": grounding_frequencies,
        }

    def _is_grounding(self, text: str, speaker: str, session_history: list[dict[str, Any]], use_history: bool) -> bool:
        system_prompt = """You are an evaluator trained to determine if a speaker’s language demonstrates grounding, which reflects active engagement and a commitment to mutual understanding in conversation. Grounding acts are characterized by clarifying questions, follow-up inquiries, or statements that seek to confirm, clarify, or expand on shared information. These acts are essential for building common ground, ensuring that both participants have a clear understanding, and preventing misunderstandings.

A grounding response often includes one or more of the following traits:

Clarifying questions: The speaker asks questions that seek clarification or further information about the other person’s statements (e.g., “Could you explain that further?” or “What did you mean by...?”).
Follow-up inquiries: The speaker shows interest in exploring a point raised by the other person, prompting them to elaborate or continue sharing (e.g., “How did that make you feel?” or “Can you tell me more about...?”).
Confirmation checks: The speaker seeks to confirm their understanding of what the other person said (e.g., “So, you mean that...?” or “Are you saying that...?”).

**Example Statements**

- "Can you tell me more about what happened at the event?"
Grounding or Not Grounding: Grounding
Reason: This is a follow-up question that prompts the other person to provide more information, demonstrating interest and a desire to deepen mutual understanding.

- "I completely understand your point."
Grounding or Not Grounding: Not Grounding
Reason: Although this statement indicates agreement, it does not actively seek further information or clarification and does not encourage continued dialogue.

- "So, you’re saying that this new policy will impact the timeline?"
Grounding or Not Grounding: Grounding
Reason: This is a confirmation check, as the speaker seeks to ensure their understanding of the other person’s statement.

- "It sounds like you’ve already made your decision."
Grounding or Not Grounding: Not Grounding
Reason: This statement reflects an observation rather than a clarifying or follow-up question, so it does not serve as a grounding act.

"""

        if use_history:
            dialogue_context = "\n".join(
                f"{turn['speaker']}: {turn['clean_text'] if 'clean_text' in turn else turn['text']}"
                for turn in session_history
            )
            dialogue_prompt = f"""Given this dialogue context:
{dialogue_context}

Determine whether the {speaker}'s last message ('{text}') is grounding or not.
Respond only with 'True' for grounding or 'False' for not grounding."""
        else:
            dialogue_prompt = f"""Determine whether the {speaker}'s last message ('{text}') is grounding or not.
Respond only with 'True' for grounding or 'False' for not grounding."""

        reflective_detection_messages = [LLMMessage(role="system", content=system_prompt + dialogue_prompt)]
        try:
            assert isinstance(self.grounding_model, OpenAIGenerator)
            response = self.grounding_model.generate(messages=reflective_detection_messages)
            return "true" in response.content.lower().strip()
        except Exception as e:
            logger.error(f"[is_grounding] {e}")
            raise ValueError("Action (is_grounding) failed.") from e
