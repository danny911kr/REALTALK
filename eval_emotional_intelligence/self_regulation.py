import re
from typing import Any, Dict, List

from tqdm import tqdm


class SelfRegulationEvaluator:
    def invoke(self, turn: str, speaker: str, session_history: List[Dict[str, Any]]) -> Any:
        # Placeholder for additional processing or API calls if needed
        pass

    def compute_score(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        sentiment_stability = {}
        emotion_consistency = {}
        sentiment_alignment = {}
        emotion_alignment = {}

        # Iterate through sessions and turns
        for session_key, session_data in tqdm(conversation.items()):
            if bool(re.match(r"^session_\d+$", session_key)):  # Filter session entries
                prev_sentiment = {}
                prev_emotion = {}

                for turn_index, turn in enumerate(session_data):
                    speaker = turn["speaker"]
                    sentiment = turn["sentiment"]
                    emotion = turn["emotion"]

                    # Initialize speaker-specific history
                    if speaker not in prev_sentiment:
                        prev_sentiment[speaker] = None
                    if speaker not in prev_emotion:
                        prev_emotion[speaker] = None
                    if speaker not in sentiment_alignment:
                        sentiment_alignment[speaker] = []
                    if speaker not in emotion_alignment:
                        emotion_alignment[speaker] = []

                    # Sentiment Stability (speaker-specific)
                    if prev_sentiment[speaker] is not None:
                        if speaker not in sentiment_stability:
                            sentiment_stability[speaker] = []
                        sentiment_stability[speaker].append(int(sentiment == prev_sentiment[speaker]))
                    prev_sentiment[speaker] = sentiment

                    # Emotion Dynamics Consistency (speaker-specific)
                    if prev_emotion[speaker] is not None:
                        if speaker not in emotion_consistency:
                            emotion_consistency[speaker] = []
                        emotion_consistency[speaker].append(int(emotion == prev_emotion[speaker]))

                    prev_emotion[speaker] = emotion

                    # Emotion and Sentiment Alignment with partner
                    if turn_index > 0:  # Compare with previous turn by the partner
                        partner_turn = session_data[turn_index - 1]
                        if partner_turn["speaker"] != speaker:  # Ensure it's the partner
                            partner_sentiment = partner_turn["sentiment"]
                            partner_emotion = partner_turn["emotion"]

                            sentiment_alignment[speaker].append(int(sentiment == partner_sentiment))
                            emotion_alignment[speaker].append(int(emotion == partner_emotion))

        # Compute final scores
        final_scores = {
            "sentiment_stability": {
                speaker: sum(scores) / len(scores) if scores else 0 for speaker, scores in sentiment_stability.items()
            },
            "emotion_consistency": {
                speaker: sum(scores) / len(scores) if scores else 0 for speaker, scores in emotion_consistency.items()
            },
            "sentiment_alignment": {
                speaker: sum(scores) / len(scores) if scores else 0 for speaker, scores in sentiment_alignment.items()
            },
            "emotion_alignment": {
                speaker: sum(scores) / len(scores) if scores else 0 for speaker, scores in emotion_alignment.items()
            },
        }

        return final_scores
