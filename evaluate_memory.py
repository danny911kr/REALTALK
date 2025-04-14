import argparse
from pathlib import Path
from typing import Any

from eval_memory.qa_tool import QATool
from utils.utils_file import read_json, write_json


def run_evaluate(data_path: str, qa_model: str, evaluate_model: str):
    input_path = Path(data_path)
    output_path = Path("output") / input_path.relative_to("data")
    result_path = output_path.parent / f"{output_path.stem}_memory_results_{qa_model}.json"
    print(f"Saving predictions to {output_path}")
    print(f"Saving results to {result_path}")

    data = read_json(data_path)
    if "conversation" in data:  # For LoCoMo data.
        qa_data = data["qa"]
        data = data["conversation"]
    else:
        qa_data = data["qa"]

    qa_tool = QATool(qa_model, evaluate_model=evaluate_model)
    results: dict[str, Any] = {
        "predictions": [],
    }
    for qa in qa_data:
        question = qa["question"]
        answer = qa.get("answer", "")
        category = qa["category"]
        conversation = data
        if category in [4, 5]:  # skip categories 4 and 5 (which are only in LoCoMo data.)
            continue
        results["predictions"].append(qa_tool.invoke(question, answer, category, conversation))

    category_scores: dict[str, dict[int, list[float]]] = {
        "lexical_score": {},
        "gpt_score": {},
    }

    for item in results["predictions"]:
        for score_type in ["lexical_score", "gpt_score"]:
            category = item["category"]
            if category not in category_scores[score_type]:
                category_scores[score_type][category] = []
            category_scores[score_type][category].append(item[score_type])

    category_averages: dict[str, dict[int, float]] = {
        score_type: {key: sum(values) / len(values) for key, values in score_dict.items()}
        for score_type, score_dict in category_scores.items()
    }

    results["lexical_mean"] = category_averages["lexical_score"]
    results["gpt_mean"] = category_averages["gpt_score"]

    print(results["lexical_mean"])
    print(results["gpt_mean"])
    write_json(results, result_path)


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/Chat_1_Emi_Elise.json")
    parser.add_argument("--qa_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--evaluate_model", type=str, default="gpt-4o-mini")

    args = parser.parse_args()
    run_evaluate(args.data_path, args.qa_model, args.evaluate_model)
