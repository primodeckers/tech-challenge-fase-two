"""Stage de pré-processamento: lê os eventos brutos e gera interações limpas."""

from src.config.params import load_params
from src.config.settings import settings
from src.data.loader import load_data
from src.data.preprocess import filter_positive_events, preprocess_data


def main() -> None:
    params = load_params()["preprocess"]

    raw_data = load_data(settings.raw_data_path)
    positive_interactions = filter_positive_events(raw_data, params["positive_events"])
    clean_interactions = preprocess_data(
        positive_interactions,
        min_user_interactions=params["min_user_interactions"],
        min_item_interactions=params["min_item_interactions"],
    )

    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.processed_data_dir / "interactions.parquet"
    clean_interactions.to_parquet(output_path, index=False)
    print(
        f"Interações limpas: {len(clean_interactions)} "
        f"({clean_interactions['user_id'].nunique()} usuários, "
        f"{clean_interactions['item_id'].nunique()} itens) -> {output_path}"
    )


if __name__ == "__main__":
    main()
