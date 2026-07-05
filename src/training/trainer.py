import pandas as pd

from src.models.base import BaseRecommender


class ModelTrainer:
    """Orquestra o treinamento de um recomendador a partir de dados prontos."""

    def __init__(self, model: BaseRecommender) -> None:
        self._model = model

    def train(self, interactions: pd.DataFrame) -> BaseRecommender:
        return self._model.fit(interactions)
