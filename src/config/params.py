from pathlib import Path
from typing import Any

import yaml

PARAMS_PATH = Path("configs/params.yaml")


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    """Carrega os hiperparâmetros do pipeline (fonte única: configs/params.yaml)."""
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)
