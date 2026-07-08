"""Protocolo de avaliação NCF: item retido de cada usuário vs. negativos amostrados."""

import numpy as np
import pandas as pd

from src.models.base import BaseRecommender


def build_candidates(
    test_data: pd.DataFrame, n_items: int, num_negatives: int, seed: int
) -> pd.DataFrame:
    """Monta, por usuário, 1 item positivo (retido) + `num_negatives` negativos."""
    rng = np.random.default_rng(seed)
    positives = test_data[["user_idx", "item_idx"]].copy()
    positives["relevant"] = 1

    sampled_users = np.repeat(test_data["user_idx"].to_numpy(), num_negatives)
    negatives = pd.DataFrame(
        {
            "user_idx": sampled_users,
            "item_idx": rng.integers(0, n_items, size=len(sampled_users)),
            "relevant": 0,
        }
    )
    return pd.concat([positives, negatives], ignore_index=True)


def score_candidates(candidates: pd.DataFrame, model: BaseRecommender) -> pd.DataFrame:
    """Aplica o modelo aos candidatos e devolve o formato esperado pelo avaliador."""
    scores = model.predict(candidates["user_idx"], candidates["item_idx"])
    return pd.DataFrame(
        {"user_id": candidates["user_idx"], "score": scores, "relevant": candidates["relevant"]}
    )
