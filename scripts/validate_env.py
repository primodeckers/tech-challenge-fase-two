"""Valida se o ambiente está pronto para rodar o pipeline (config + dados)."""

import sys

from src.config.settings import settings


def main() -> None:
    problems = _collect_problems()
    if problems:
        print("Ambiente inválido:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print("Ambiente OK.")


def _collect_problems() -> list[str]:
    problems = []
    if not settings.mlflow_tracking_uri:
        problems.append("MLFLOW_TRACKING_URI não configurado")
    if not settings.raw_data_path.parent.exists():
        problems.append(f"pasta de dados brutos não existe: {settings.raw_data_path.parent}")
    if settings.random_seed < 0:
        problems.append("RANDOM_SEED deve ser um inteiro não-negativo")
    return problems


if __name__ == "__main__":
    main()
