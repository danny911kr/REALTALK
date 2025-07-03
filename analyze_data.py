import argparse
from pathlib import Path
import re

from tqdm import tqdm

from eval_emotional_intelligence import (
    EmpathyEvaluator,
    MotivationEvaluator,
    SelfAwarenessEvaluator,
    SelfRegulationEvaluator,
    SocialSkillsEvaluator,
)
from utils.utils_file import read_json, write_json


def run_evaluate(evaluate_config, evaluate_modes):
    input_path = Path(evaluate_config["data_path"])
    output_path = Path("output") / input_path.relative_to("data")
    result_path = output_path.parent / f"{output_path.stem}_results.json"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving predictions to {output_path}")
    print(f"Saving results to {result_path}")

    data = read_json(evaluate_config["data_path"])
    if "conversation" in data:  # For LoCoMo data.
        data = data["conversation"]

    sa_evaluator = SelfAwarenessEvaluator(
        reflective_model=evaluate_config.get("reflective_model"),
        sentiment_model=evaluate_config.get("sentiment_model"),
        emotion_model=evaluate_config.get("emotion_model"),
    )
    motivation_evaluator = MotivationEvaluator(grounding_model=evaluate_config.get("grounding_model"))
    social_skills_evaluator = SocialSkillsEvaluator(intimacy_model=evaluate_config.get("intimacy_model"))
    empathy_evaluator = EmpathyEvaluator(empathy_model=evaluate_config.get("empathy_model"))
    self_regulation_evaluator = SelfRegulationEvaluator()

    # Evaluate each turn
    for session_key, session_data in tqdm(data.items()):
        if bool(re.match(r"^session_\d+$", session_key)):
            for i, turn in enumerate(session_data):
                speaker = turn["speaker"]
                text = turn.get("clean_text", turn.get("text", ""))

                if "self-awareness" in evaluate_modes:
                    result = sa_evaluator.invoke(text, speaker, session_data[: i + 1])
                    turn.update(result)

                if "motivation" in evaluate_modes:
                    result = motivation_evaluator.invoke(text, speaker, session_data[: i + 1])
                    turn.update(result)

                if "social-skills" in evaluate_modes:
                    result = social_skills_evaluator.invoke(text, speaker, session_data[: i + 1])
                    turn.update(result)

                if "empathy" in evaluate_modes:
                    result = empathy_evaluator.invoke(text, speaker, session_data[: i + 1])
                    turn.update(result)

    # Aggregate results
    results = {}
    if "self-awareness" in evaluate_modes:
        sa_results = sa_evaluator.compute_score(data)
        results.update(sa_results)
        print("Self-awareness results:")
        print(sa_results)
    if "motivation" in evaluate_modes:
        motivation_results = motivation_evaluator.compute_score(data)
        results.update(motivation_results)
        print("Motivation results:")
        print(motivation_results)
    if "social-skills" in evaluate_modes:
        social_skills_results = social_skills_evaluator.compute_score(data)
        results.update(social_skills_results)
        print("Social skills results:")
        print(social_skills_results)
    if "empathy" in evaluate_modes:
        empathy_results = empathy_evaluator.compute_score(data)
        results.update(empathy_results)
        print("Empathy results:")
        print(empathy_results)
    if "self-regulation" in evaluate_modes:
        self_regulation_results = self_regulation_evaluator.compute_score(data)
        results.update(self_regulation_results)
        print("Self-regulation results:")
        print(self_regulation_results)

    write_json(data, output_path)
    write_json(results, result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/Chat_1_Emi_Elise.json")
    parser.add_argument("--reflective_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--empathy_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--grounding_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--sentiment_model", type=str, default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    parser.add_argument("--emotion_model", type=str, default="cardiffnlp/twitter-roberta-large-emotion-latest")
    parser.add_argument("--intimacy_model", type=str, default="cardiffnlp/twitter-roberta-large-intimacy-latest")
    args = parser.parse_args()

    config = {
        "data_path": args.data_path,
        "reflective_model": args.reflective_model,
        "sentiment_model": args.sentiment_model,
        "emotion_model": args.emotion_model,
        "grounding_model": args.grounding_model,
        "intimacy_model": args.intimacy_model,
        "empathy_model": args.empathy_model,
    }
    evaluate_modes = ["self-awareness", "empathy", "motivation", "social-skills", "self-regulation"]
    run_evaluate(config, evaluate_modes)
