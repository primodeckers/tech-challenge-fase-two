import numpy as np
import pandas as pd


class RankingEvaluator:
    """Calcula métricas de qualidade de ranking (precision, recall, ndcg, hit rate) @k."""

    def __init__(self, k: int = 10) -> None:
        self._k = k

    def evaluate(self, predictions: pd.DataFrame) -> dict[str, float]:
        """predictions: colunas user_id, item_id, score, relevant (0/1)."""
        per_user = predictions.groupby("user_id").apply(self._rank_user, include_groups=False)
        metrics = pd.DataFrame(per_user.tolist(), index=per_user.index)
        return metrics.mean().to_dict()

    def _rank_user(self, group: pd.DataFrame) -> dict[str, float]:
        top_k = group.sort_values("score", ascending=False).head(self._k)
        n_relevant_total = group["relevant"].sum()
        hits = top_k["relevant"].to_numpy()
        return {
            "precision_at_k": hits.sum() / self._k,
            "recall_at_k": hits.sum() / n_relevant_total if n_relevant_total > 0 else 0.0,
            "ndcg_at_k": self._ndcg(hits),
            "hit_rate_at_k": float(hits.sum() > 0),
        }

    @staticmethod
    def _ndcg(hits: np.ndarray) -> float:
        discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
        dcg = (hits * discounts).sum()
        ideal_hits = np.sort(hits)[::-1]
        idcg = (ideal_hits * discounts).sum()
        return dcg / idcg if idcg > 0 else 0.0
