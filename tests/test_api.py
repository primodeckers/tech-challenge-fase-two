import json

import torch
from fastapi.testclient import TestClient

from src.api import app as api_module
from src.models.factory import ModelFactory


def _prepare_model(tmp_path, monkeypatch):
    """Cria um modelo e metadados pequenos em disco e aponta os settings para lá."""
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    processed_dir.mkdir()
    models_dir.mkdir()

    metadata = {"n_users": 5, "n_items": 8}
    (processed_dir / "metadata.json").write_text(json.dumps(metadata))
    model = ModelFactory.create("mlp", n_users=5, n_items=8, embedding_dim=16)
    torch.save(model.state_dict, models_dir / "model.pt")

    monkeypatch.setattr(api_module.settings, "processed_data_dir", processed_dir)
    monkeypatch.setattr(api_module.settings, "model_output_dir", models_dir)
    return metadata


def test_health_and_recommend_endpoints(tmp_path, monkeypatch):
    metadata = _prepare_model(tmp_path, monkeypatch)

    with TestClient(api_module.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        response = client.get("/recomendar/0", params={"k": 3})
        assert response.status_code == 200
        body = response.json()
        assert body["user_idx"] == 0
        assert len(body["recomendacoes"]) == 3
        # itens recomendados são índices válidos e vêm ordenados por score.
        scores = [rec["score"] for rec in body["recomendacoes"]]
        assert scores == sorted(scores, reverse=True)
        assert all(0 <= rec["item_idx"] < metadata["n_items"] for rec in body["recomendacoes"])
