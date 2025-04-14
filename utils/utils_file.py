import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union


def create_dir(dirname: Path) -> None:
    if not dirname.exists():
        dirname.mkdir(parents=True)


def read_json(filepath: Union[str, Path]) -> Dict[Any, Any]:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(filepath: Union[str, Path]) -> List[Dict[Any, Any]]:
    if not os.path.exists(filepath):
        return []

    json_lines = []
    with open(filepath, encoding="utf-8") as f:
        while True:
            file_line = f.readline().strip()
            if not file_line:
                break
            json_lines.append(json.loads(file_line))
    return json_lines


def write_json(data: Union[Dict[Any, Any], List[Dict[Any, Any]]], filepath: Union[str, Path]) -> None:
    dirname = Path(filepath).absolute().parent
    create_dir(dirname)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def write_jsonl(data: List[Dict[Any, Any]], filepath: Union[str, Path]) -> None:
    dirname = Path(filepath).absolute().parent
    create_dir(dirname)
    with open(filepath, "w", encoding="utf-8") as f:
        for datum in data:
            f.write(json.dumps(datum, ensure_ascii=False) + "\n")
