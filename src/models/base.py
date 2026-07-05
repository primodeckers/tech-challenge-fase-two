from abc import ABC, abstractmethod

import pandas as pd


class BaseRecommender(ABC):
    """Contrato comum a todos os modelos de recomendação (baselines e rede neural)."""

    @abstractmethod
    def fit(self, interactions: pd.DataFrame) -> "BaseRecommender":
        """Treina o modelo a partir de interações user-item."""

    @abstractmethod
    def predict(self, user_ids: pd.Series, item_ids: pd.Series) -> pd.Series:
        """Retorna o score de afinidade previsto para pares (user, item)."""
