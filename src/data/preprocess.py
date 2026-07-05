import pandas as pd


def preprocess_data(raw_data: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame:
    """Remove duplicatas e filtra usuários/itens com poucas interações."""
    processed_data = raw_data.drop_duplicates(subset=["user_id", "item_id", "timestamp"])
    processed_data = _filter_by_min_interactions(processed_data, "user_id", min_interactions)
    processed_data = _filter_by_min_interactions(processed_data, "item_id", min_interactions)
    return processed_data.reset_index(drop=True)


def _filter_by_min_interactions(data: pd.DataFrame, column: str, min_count: int) -> pd.DataFrame:
    counts = data[column].value_counts()
    valid_ids = counts[counts >= min_count].index
    return data[data[column].isin(valid_ids)]
