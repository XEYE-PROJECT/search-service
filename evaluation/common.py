"""Rutas y utilidades compartidas por evaluate y compare."""

import json
import re
import unicodedata
from pathlib import Path

EVALUATION_DIR = Path(__file__).resolve().parent
DATASETS_DIR = EVALUATION_DIR / "datasets"
RESULTS_DIR = EVALUATION_DIR / "results"


def slugify(name: str) -> str:
    text = unicodedata.normalize("NFD", name.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "list"


def list_results_dir(list_name: str) -> Path:
    return RESULTS_DIR / slugify(list_name)


def dataset_path(list_name: str) -> Path:
    return DATASETS_DIR / f"{slugify(list_name)}.json"


def load_runs(list_name: str) -> list[dict]:
    """Todos los runs guardados de una lista, ordenados por timestamp ascendente."""
    directory = list_results_dir(list_name)
    if not directory.is_dir():
        return []
    runs = []
    for path in sorted(directory.glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            run = json.load(fh)
        run["_file"] = path.name
        runs.append(run)
    runs.sort(key=lambda r: r.get("timestamp", ""))
    return runs
