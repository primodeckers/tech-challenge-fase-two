import pandas as pd

from src.models.neural import MLPRecommender


def _tiny_interactions() -> pd.DataFrame:
    return pd.DataFrame({"user_idx": [0, 1, 2, 0, 1], "item_idx": [0, 1, 2, 1, 2]})


def test_mlp_trains_and_predicts_scores():
    model = MLPRecommender(n_users=3, n_items=3, epochs=2, batch_size=4, seed=0)
    model.fit(_tiny_interactions())

    scores = model.predict(pd.Series([0, 1]), pd.Series([0, 1]))

    assert len(scores) == 2
    assert scores.between(0.0, 1.0).all()  # saída sigmoide


def test_mlp_early_stopping_invokes_callback_per_epoch():
    calls: list[int] = []
    model = MLPRecommender(n_users=3, n_items=3, epochs=5, patience=1, batch_size=4, seed=0)
    model.fit(
        _tiny_interactions(),
        val_interactions=_tiny_interactions(),
        epoch_callback=lambda epoch, train_loss, val_loss: calls.append(epoch),
    )

    assert len(calls) >= 1  # ao menos uma época executada
    assert calls == sorted(calls)  # épocas em ordem crescente
