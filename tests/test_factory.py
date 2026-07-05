import pytest

from src.models.baselines import PopularityRecommender
from src.models.factory import ModelFactory
from src.models.neural import MLPRecommender


def test_factory_creates_popularity_model():
    model = ModelFactory.create("popularity")
    assert isinstance(model, PopularityRecommender)


def test_factory_creates_mlp_model():
    model = ModelFactory.create("mlp", n_users=10, n_items=20)
    assert isinstance(model, MLPRecommender)


def test_factory_raises_for_unknown_model():
    with pytest.raises(ValueError, match="Modelo desconhecido"):
        ModelFactory.create("modelo-inexistente")
