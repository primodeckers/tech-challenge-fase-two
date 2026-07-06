import pandas as pd

from src.data.preprocess import filter_positive_events, preprocess_data


def test_filter_positive_events_keeps_only_listed_events():
    data = pd.DataFrame(
        {"user_id": [1, 2, 3], "item_id": ["a", "b", "c"], "event": ["view", "click", "buy"]}
    )
    result = filter_positive_events(data, ["view", "buy"])

    assert set(result["event"]) == {"view", "buy"}
    assert len(result) == 2


def test_preprocess_removes_duplicates_and_filters_by_min_interactions():
    data = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2],
            "item_id": ["a", "a", "a", "a", "b"],
            "timestamp": [10, 10, 20, 30, 40],
        }
    )
    result = preprocess_data(data, min_user_interactions=2, min_item_interactions=2)

    # A duplicata exata (user 1, item a, ts 10) é removida; usuário 2 e item b
    # ficam abaixo do mínimo e são descartados.
    assert len(result) == 3
    assert set(result["user_id"]) == {1}
    assert set(result["item_id"]) == {"a"}
