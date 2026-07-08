"""Runner de experimentos: treina várias configurações, registra cada uma no
MLflow e promove a melhor (por NDCG) a produção no Model Registry."""

import json

import mlflow
import pandas as pd
from mlflow import MlflowClient

from src.config.params import load_params
from src.config.settings import settings
from src.evaluation.evaluator import RankingEvaluator
from src.evaluation.protocol import build_candidates, score_candidates
from src.models.baselines import PopularityRecommender
from src.models.factory import ModelFactory


def main() -> None:
    params = load_params()
    base_params, eval_params = params["train"], params["evaluate"]
    exp_params = params["experiment"]

    processed_dir = settings.processed_data_dir
    train_data = pd.read_parquet(processed_dir / "train.parquet")
    val_data = pd.read_parquet(processed_dir / "val.parquet")
    test_data = pd.read_parquet(processed_dir / "test.parquet")
    metadata = json.loads((processed_dir / "metadata.json").read_text())

    evaluator = RankingEvaluator(eval_params["k"])
    candidates = build_candidates(
        test_data, metadata["n_items"], eval_params["num_eval_negatives"], settings.random_seed
    )
    baseline = PopularityRecommender().fit(train_data.rename(columns={"item_idx": "item_id"}))
    baseline_metrics = evaluator.evaluate(score_candidates(candidates, baseline))

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    results = [
        _run_config(
            config,
            base_params,
            metadata,
            train_data,
            val_data,
            candidates,
            evaluator,
            baseline_metrics,
            exp_params["registered_model_name"],
        )
        for config in exp_params["configs"]
    ]
    _promote_best(results, exp_params["registered_model_name"], exp_params["production_alias"])


def _run_config(
    config: dict,
    base_params: dict,
    metadata: dict,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    candidates: pd.DataFrame,
    evaluator: RankingEvaluator,
    baseline_metrics: dict,
    registered_model_name: str,
) -> dict:
    hyperparams = {**base_params, **config}
    with mlflow.start_run(run_name=f"mlp_emb{config['embedding_dim']}"):
        mlflow.set_tag("model_type", "mlp")
        mlflow.log_params(hyperparams)

        model = ModelFactory.create(
            "mlp",
            n_users=metadata["n_users"],
            n_items=metadata["n_items"],
            embedding_dim=hyperparams["embedding_dim"],
            learning_rate=hyperparams["learning_rate"],
            epochs=hyperparams["epochs"],
            patience=hyperparams["patience"],
            negative_ratio=hyperparams["negative_ratio"],
            seed=settings.random_seed,
        )
        model.fit(train_data, val_data, epoch_callback=_log_epoch)

        metrics = evaluator.evaluate(score_candidates(candidates, model))
        mlflow.log_metrics({f"test_{name}": value for name, value in metrics.items()})
        mlflow.log_metrics({f"baseline_{name}": value for name, value in baseline_metrics.items()})

        model_info = mlflow.pytorch.log_model(
            model.torch_module,
            name="model",
            registered_model_name=registered_model_name,
            serialization_format="pickle",
        )
        _print_result(config, metrics)
        return {"version": model_info.registered_model_version, "ndcg": metrics["ndcg_at_k"]}


def _promote_best(results: list[dict], model_name: str, alias: str) -> None:
    best = max(results, key=lambda result: result["ndcg"])
    client = MlflowClient()
    client.set_registered_model_alias(model_name, alias, best["version"])
    client.set_model_version_tag(model_name, best["version"], "status", "production")
    print(
        f"Melhor modelo: versão {best['version']} (NDCG@10={best['ndcg']:.4f}) "
        f"promovido a produção (alias '@{alias}')."
    )


def _log_epoch(epoch: int, train_loss: float, val_loss: float) -> None:
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)


def _print_result(config: dict, metrics: dict) -> None:
    formatted = ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    print(f"[emb={config['embedding_dim']}] {formatted}")


if __name__ == "__main__":
    main()
