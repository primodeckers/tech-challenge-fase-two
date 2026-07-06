import pandas as pd

from src.features.split import leave_one_out_split


def test_leave_one_out_holds_most_recent_per_user():
    interactions = pd.DataFrame(
        {
            "user_idx": [0, 0, 0, 0, 1, 1, 1],
            "item_idx": [10, 11, 12, 13, 20, 21, 22],
            "timestamp": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    train, val, test = leave_one_out_split(interactions)

    # A interação mais recente de cada usuário vai para teste; a 2ª mais recente, val.
    assert test.set_index("user_idx")["item_idx"].to_dict() == {0: 13, 1: 22}
    assert val.set_index("user_idx")["item_idx"].to_dict() == {0: 12, 1: 21}
    # Todo usuário de teste também aparece no treino.
    assert set(test["user_idx"]).issubset(set(train["user_idx"]))
