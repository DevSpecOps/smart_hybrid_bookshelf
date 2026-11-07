import json
from pathlib import Path


def save_metrics(results, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return p


def load_metrics(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
