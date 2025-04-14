from collections import Counter
import os
import string
from typing import Any

from loguru import logger
from nltk.stem import PorterStemmer
import numpy as np
import regex

from utils.utils_llm import LLMMessage, OpenAIGenerator


ps = PorterStemmer()


class QATool:
    def __init__(self, qa_model, evaluate_model):
        self.qa_model = OpenAIGenerator(
            model=qa_model,
            oai_api_key=os.getenv("OPENAI_API_KEY", ""),
        )
        self.evaluate_model = OpenAIGenerator(
            model=evaluate_model,
            oai_api_key=os.getenv("OPENAI_API_KEY", ""),
        )
        self.CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is written at the beginning of the conversation.\n\n"
        self.QA_PROMPT = """Based on the above context, write an answer in the form of a short phrase for the following question.
If the question is about a date, try to infer the approximate date (e.g., "In the 1800s", "Before Jan 2021", etc.).

Question: {}
Answer:
"""
        self.QA_PROMPT_CAT_5 = """Based on the above context, answer the following question.
Question: {}
Answer:
"""

    def invoke(self, question: str, answer: str, category: int, conversation: dict[str, Any]) -> Any:
        # start instruction prompt
        speakers_names = list(set([d["speaker"] for d in conversation["session_1"]]))
        start_prompt = self.CONV_START_PROMPT.format(speakers_names[0], speakers_names[1])
        query_conv = self.get_input_context(conversation)
        query_conv = start_prompt + query_conv

        query = query_conv + "\n\n" + self.QA_PROMPT.format(question)

        # generate answer
        qa_messages = [LLMMessage(role="user", content=query)]
        try:
            response = self.qa_model.generate(messages=qa_messages)
        except Exception as e:
            logger.error(f"[qa] {e}")
            raise ValueError("Action (qa) failed.") from e

        output = response.content
        score = self.compute_lexical_score(output, answer, category)  # f1 score
        gpt_score = self.gpt_score(question, output, answer)

        return {
            "question": question,
            "answer": answer,
            "category": category,
            "prediction": response.content,
            "lexical_score": score,
            "gpt_score": gpt_score,
        }

    def get_input_context(self, conversation):
        query_conv = ""
        session_nums = sorted(
            int(k.split("_")[-1]) for k in conversation.keys() if "session" in k and "date_time" not in k
        )

        for i in session_nums:
            session_key = f"session_{i}"
            if session_key in conversation:
                session_date = f"DATE: {conversation[f'{session_key}_date_time']}\n"
                conversation_str = ""

                for dialog in reversed(conversation[session_key]):
                    turn = f'{dialog["speaker"]} said, "{dialog["clean_text"] if "clean_text" in dialog else dialog["text"]}"'
                    if "blip_caption" in dialog:
                        turn += f'\n{dialog["speaker"]} shared, an image of "{dialog["blip_caption"]}".'
                    turn += "\n"
                    conversation_str = turn + conversation_str

                query_conv = session_date + "CONVERSATION:\n" + conversation_str + "\n" + query_conv
        return query_conv

    def normalize_answer(self, s):
        s = s.replace(",", "")

        def remove_articles(text):
            # return regex.sub(r'\b(a|an|the)\b', ' ', text)
            return regex.sub(r"\b(a|an|the|and)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def f1_score(self, prediction, ground_truth):
        prediction_tokens = [ps.stem(w) for w in self.normalize_answer(prediction).split()]
        ground_truth_tokens = [ps.stem(w) for w in self.normalize_answer(ground_truth).split()]
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def f1(self, prediction, ground_truth) -> float:
        predictions = [p.strip() for p in prediction.split(",")]
        ground_truths = [g.strip() for g in ground_truth.split(",")]

        return float(
            np.mean([max([self.f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths])
        )

    def compute_lexical_score(self, prediction: str, answer: str, category: int) -> float:
        print("prediction:", prediction)
        if category == 3:
            answer = answer.split(";")[0].strip()

        if category in [2, 3, 4]:
            score = self.f1_score(str(prediction), str(answer))
        elif category in [1]:
            score = self.f1(str(prediction), str(answer))
        elif category in [5]:
            if "no information available" in prediction.lower() or "not mentioned" in prediction.lower():
                score = 1.0
            else:
                score = 0.0
        else:
            raise ValueError(f"Invalid category: {category}")

        return score

    def gpt_score(self, question, prediction, answer):
        evaluate_prompt = f"""Given the question and its ground truth answer, evaluate the correctness of the model's prediction.

Question: {question}
Ground truth: {answer}
Model's prediction: {prediction}

Assign a score between 0 and 1, where 0 indicates the model's prediction is completely incorrect, and 1 indicates the model's prediction is completely correct.
Output in following JSON format:
{{
    "score": <score>,
}}
"""
        qa_messages = [LLMMessage(role="user", content=evaluate_prompt)]
        try:
            response = self.qa_model.generate_json(messages=qa_messages)
            score = response.content["score"]
        except Exception as e:
            logger.error(f"[gpt_score] {e}")
            raise ValueError("Action (gpt_score) failed.") from e
        return score
