import pandas as pd

from src.pipeline.evaluate import _build_candidates


def test_build_candidates_has_one_positive_and_n_negatives_per_user():
    test_data = pd.DataFrame({"user_idx": [0, 1], "item_idx": [5, 7]})

    candidates = _build_candidates(test_data, n_items=100, num_negatives=3, seed=0)

    # 2 positivos (1 por usuário) + 2*3 negativos = 8 candidatos.
    assert len(candidates) == 8
    assert candidates["relevant"].sum() == 2
    # O item retido de cada usuário aparece como positivo.
    positives = candidates[candidates["relevant"] == 1]
    assert positives.set_index("user_idx")["item_idx"].to_dict() == {0: 5, 1: 7}
    # Cada usuário tem exatamente num_negatives candidatos negativos.
    negatives = candidates[candidates["relevant"] == 0]
    assert (negatives["user_idx"].value_counts() == 3).all()
