import pandas as pd
import pytest

from src.models.baselines import PopularityRecommender


def test_popularity_recommender_scores_frequent_items_higher():
    interactions = pd.DataFrame({"user_id": [1, 1, 2, 3], "item_id": ["x", "x", "x", "y"]})
    model = PopularityRecommender().fit(interactions)

    scores = model.predict(pd.Series([1, 1]), pd.Series(["x", "y"]))

    assert scores.iloc[0] > scores.iloc[1]


def test_popularity_recommender_raises_if_not_fitted():
    model = PopularityRecommender()
    with pytest.raises(RuntimeError):
        model.predict(pd.Series([1]), pd.Series(["x"]))
