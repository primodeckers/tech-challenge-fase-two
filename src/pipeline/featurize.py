"""Stage de featurização: codifica IDs e faz o split leave-one-out por usuário."""

import json

import pandas as pd

from src.config.settings import settings
from src.features.encoding import IdEncoder
from src.features.split import leave_one_out_split

_SPLIT_COLUMNS = ["user_idx", "item_idx", "timestamp"]


def main() -> None:
    processed_dir = settings.processed_data_dir
    interactions = pd.read_parquet(processed_dir / "interactions.parquet")

    user_encoder = IdEncoder().fit(interactions["user_id"])
    item_encoder = IdEncoder().fit(interactions["item_id"])
    interactions["user_idx"] = user_encoder.transform(interactions["user_id"])
    interactions["item_idx"] = item_encoder.transform(interactions["item_id"])

    train, val, test = leave_one_out_split(interactions)
    _save_split(train, processed_dir / "train.parquet")
    _save_split(val, processed_dir / "val.parquet")
    _save_split(test, processed_dir / "test.parquet")

    metadata = {"n_users": user_encoder.n_unique, "n_items": item_encoder.n_unique}
    (processed_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(
        f"Split leave-one-out -> treino: {len(train)}, validação: {len(val)}, "
        f"teste: {len(test)} | usuários: {metadata['n_users']}, itens: {metadata['n_items']}"
    )


def _save_split(split: pd.DataFrame, path) -> None:
    split[_SPLIT_COLUMNS].to_parquet(path, index=False)


if __name__ == "__main__":
    main()
