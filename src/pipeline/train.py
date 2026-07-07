"""Stage de treino: treina a rede neural e registra o experimento no MLflow."""

import json

import mlflow
import pandas as pd
import torch

from src.config.params import load_params
from src.config.settings import settings
from src.models.factory import ModelFactory


def main() -> None:
    train_params = load_params()["train"]
    train_data = pd.read_parquet(settings.processed_data_dir / "train.parquet")
    val_data = pd.read_parquet(settings.processed_data_dir / "val.parquet")
    metadata = json.loads((settings.processed_data_dir / "metadata.json").read_text())

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="mlp") as run:
        mlflow.set_tag("model_type", "mlp")
        mlflow.log_params(train_params)
        mlflow.log_params({"n_users": metadata["n_users"], "n_items": metadata["n_items"]})

        model = ModelFactory.create(
            "mlp",
            n_users=metadata["n_users"],
            n_items=metadata["n_items"],
            embedding_dim=train_params["embedding_dim"],
            learning_rate=train_params["learning_rate"],
            epochs=train_params["epochs"],
            patience=train_params["patience"],
            negative_ratio=train_params["negative_ratio"],
            seed=settings.random_seed,
        )
        model.fit(train_data, val_data, epoch_callback=_log_epoch)

        settings.model_output_dir.mkdir(parents=True, exist_ok=True)
        model_path = settings.model_output_dir / "model.pt"
        torch.save(model.state_dict, model_path)
        mlflow.log_artifact(str(model_path))

        # Guarda o run_id para o stage de avaliação registrar as métricas de
        # teste no MESMO run (linhagem: curva de loss + métricas finais juntas).
        (settings.model_output_dir / "run_id.txt").write_text(run.info.run_id)
        print(f"Modelo treinado salvo em {model_path} (run {run.info.run_id})")


def _log_epoch(epoch: int, train_loss: float, val_loss: float) -> None:
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)


if __name__ == "__main__":
    main()
