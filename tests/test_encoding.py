import pandas as pd

from src.features.encoding import IdEncoder


def test_id_encoder_maps_to_contiguous_indices():
    ids = pd.Series(["a", "b", "a", "c"])
    encoder = IdEncoder().fit(ids)

    assert encoder.n_unique == 3
    assert set(encoder.transform(ids).unique()) == {0, 1, 2}
