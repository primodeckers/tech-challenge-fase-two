import pandas as pd

from src.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Recomenda por popularidade global do item — baseline simples."""

    def __init__(self) -> None:
        self._item_scores: pd.Series | None = None

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        counts = interactions["item_id"].value_counts()
        self._item_scores = counts / counts.max()
        return self

    def predict(self, user_ids: pd.Series, item_ids: pd.Series) -> pd.Series:
        if self._item_scores is None:
            raise RuntimeError("Modelo não treinado: chame fit() antes de predict().")
        return item_ids.map(self._item_scores).fillna(0.0)
