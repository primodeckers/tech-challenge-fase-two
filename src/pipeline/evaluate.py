"""Stage de avaliação: ranqueia o item retido de cada usuário contra negativos."""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

from src.config.params import load_params
from src.config.settings import settings
from src.evaluation.evaluator import RankingEvaluator
from src.models.base import BaseRecommender
from src.models.baselines import PopularityRecommender
from src.models.neural import MLPRecommender

_METRICS_PATH = Path("metrics/metrics.json")


def main() -> None:
    params = load_params()
    train_params, eval_params = params["train"], params["evaluate"]

    processed_dir = settings.processed_data_dir
    train_data = pd.read_parquet(processed_dir / "train.parquet")
    test_data = pd.read_parquet(processed_dir / "test.parquet")
    metadata = json.loads((processed_dir / "metadata.json").read_text())

    candidates = _build_candidates(
        test_data, metadata["n_items"], eval_params["num_eval_negatives"], settings.random_seed
    )
    evaluator = RankingEvaluator(eval_params["k"])

    mlp = _load_mlp(metadata, train_params["embedding_dim"])
    baseline = PopularityRecommender().fit(train_data.rename(columns={"item_idx": "item_id"}))

    metrics = {
        "mlp": evaluator.evaluate(_score(candidates, mlp)),
        "baseline_popularity": evaluator.evaluate(_score(candidates, baseline)),
    }
    _save_metrics(metrics)
    _log_to_mlflow(metrics)
    _print_summary(metrics, eval_params["k"])


def _build_candidates(
    test_data: pd.DataFrame, n_items: int, num_negatives: int, seed: int
) -> pd.DataFrame:
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


def _load_mlp(metadata: dict, embedding_dim: int) -> MLPRecommender:
    model = MLPRecommender(
        n_users=metadata["n_users"], n_items=metadata["n_items"], embedding_dim=embedding_dim
    )
    state_dict = torch.load(settings.model_output_dir / "model.pt")
    return model.load_state(state_dict)


def _score(candidates: pd.DataFrame, model: BaseRecommender) -> pd.DataFrame:
    scores = model.predict(candidates["user_idx"], candidates["item_idx"])
    return pd.DataFrame(
        {"user_id": candidates["user_idx"], "score": scores, "relevant": candidates["relevant"]}
    )


def _save_metrics(metrics: dict) -> None:
    _METRICS_PATH.parent.mkdir(exist_ok=True)
    _METRICS_PATH.write_text(json.dumps(metrics, indent=2))


def _log_to_mlflow(metrics: dict) -> None:
    run_id_path = settings.model_output_dir / "run_id.txt"
    if not run_id_path.exists():
        return
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    with mlflow.start_run(run_id=run_id_path.read_text().strip()):
        for model_name, model_metrics in metrics.items():
            for metric_name, value in model_metrics.items():
                mlflow.log_metric(f"{model_name}_{metric_name}", value)


def _print_summary(metrics: dict, k: int) -> None:
    for model_name, model_metrics in metrics.items():
        formatted = ", ".join(f"{name}={value:.4f}" for name, value in model_metrics.items())
        print(f"[@{k}] {model_name}: {formatted}")


if __name__ == "__main__":
    main()
