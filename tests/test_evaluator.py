import pandas as pd

from src.evaluation.evaluator import RankingEvaluator


def test_evaluator_computes_metrics_for_perfect_ranking():
    predictions = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "item_id": ["a", "b", "c"],
            "score": [0.9, 0.5, 0.1],
            "relevant": [1, 0, 0],
        }
    )
    metrics = RankingEvaluator(k=1).evaluate(predictions)

    assert metrics["precision_at_k"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["hit_rate_at_k"] == 1.0
    assert metrics["ndcg_at_k"] == 1.0


def test_evaluator_zero_when_no_relevant_items():
    predictions = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": ["a", "b"],
            "score": [0.9, 0.5],
            "relevant": [0, 0],
        }
    )
    metrics = RankingEvaluator(k=1).evaluate(predictions)

    assert metrics["recall_at_k"] == 0.0
    assert metrics["hit_rate_at_k"] == 0.0
