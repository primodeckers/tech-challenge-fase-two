import pandas as pd


class IdEncoder:
    """Mapeia IDs originais (usuário ou item) para índices contíguos 0..n-1."""

    def __init__(self) -> None:
        self._id_to_index: dict[object, int] = {}

    def fit(self, ids: pd.Series) -> "IdEncoder":
        unique_ids = ids.unique()
        self._id_to_index = {id_: idx for idx, id_ in enumerate(unique_ids)}
        return self

    def transform(self, ids: pd.Series) -> pd.Series:
        return ids.map(self._id_to_index)

    @property
    def n_unique(self) -> int:
        return len(self._id_to_index)
