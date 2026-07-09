"""API de inferência: serve o recomendador treinado via HTTP (FastAPI)."""

import json
from contextlib import asynccontextmanager

import pandas as pd
import torch
from fastapi import FastAPI

from src.config.settings import settings
from src.models.neural import MLPRecommender

_state: dict = {}


def _load_model() -> None:
    metadata = json.loads((settings.processed_data_dir / "metadata.json").read_text())
    state_dict = torch.load(settings.model_output_dir / "model.pt", map_location="cpu")
    # A dimensão do embedding é inferida do próprio peso salvo (sem depender de params).
    embedding_dim = state_dict["user_embedding.weight"].shape[1]

    model = MLPRecommender(
        n_users=metadata["n_users"], n_items=metadata["n_items"], embedding_dim=embedding_dim
    )
    model.load_state(state_dict)
    _state["model"] = model
    _state["n_items"] = metadata["n_items"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield
    _state.clear()


app = FastAPI(title="Recomendador de Produtos", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in _state}


@app.get("/recomendar/{user_idx}")
def recomendar(user_idx: int, k: int = 10) -> dict:
    """Retorna os top-k itens recomendados para um usuário (por índice)."""
    model, n_items = _state["model"], _state["n_items"]
    users = pd.Series([user_idx] * n_items)
    items = pd.Series(range(n_items))
    scores = model.predict(users, items)
    top_items = scores.nlargest(k)
    return {
        "user_idx": user_idx,
        "recomendacoes": [
            {"item_idx": int(item_idx), "score": round(float(score), 4)}
            for item_idx, score in top_items.items()
        ],
    }
