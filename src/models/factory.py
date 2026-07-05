from typing import Any

from src.models.base import BaseRecommender
from src.models.baselines import PopularityRecommender
from src.models.neural import MLPRecommender


class ModelFactory:
    """Cria instâncias de recomendadores a partir de um nome (Factory Method)."""

    _registry: dict[str, type[BaseRecommender]] = {
        "popularity": PopularityRecommender,
        "mlp": MLPRecommender,
    }

    @classmethod
    def create(cls, model_name: str, **kwargs: Any) -> BaseRecommender:
        model_class = cls._registry.get(model_name)
        if model_class is None:
            raise ValueError(f"Modelo desconhecido: '{model_name}'. Opções: {list(cls._registry)}")
        return model_class(**kwargs)
