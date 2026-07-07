import copy
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models.base import BaseRecommender

# Callback chamado ao fim de cada época: (época, loss_treino, loss_validação).
EpochCallback = Callable[[int, float, float], None]


class _EmbeddingMLP(nn.Module):
    """Rede embedding-based (estilo Neural Collaborative Filtering)."""

    def __init__(self, n_users: int, n_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        embeddings = torch.cat(
            [self.user_embedding(user_idx), self.item_embedding(item_idx)], dim=1
        )
        return torch.sigmoid(self.mlp(embeddings)).squeeze(-1)


class MLPRecommender(BaseRecommender):
    """Recomendador neural com embeddings de usuário/item.

    Treina em mini-batches com amostragem negativa (reamostrada a cada época) e
    faz early stopping pela perda de validação, restaurando o melhor estado.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 32,
        learning_rate: float = 1e-3,
        epochs: int = 20,
        patience: int = 3,
        negative_ratio: int = 1,
        batch_size: int = 2048,
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        self._model = _EmbeddingMLP(n_users, n_items, embedding_dim)
        self._n_items = n_items
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._patience = patience
        self._negative_ratio = negative_ratio
        self._batch_size = batch_size
        self._rng = np.random.default_rng(seed)

    def fit(
        self,
        interactions: pd.DataFrame,
        val_interactions: pd.DataFrame | None = None,
        epoch_callback: EpochCallback | None = None,
    ) -> "MLPRecommender":
        """Treina o modelo. Espera colunas 'user_idx' e 'item_idx'."""
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
        loss_fn = nn.BCELoss()
        val_tensors = (
            self._build_tensors(val_interactions) if val_interactions is not None else None
        )

        best_val_loss = float("inf")
        best_state = copy.deepcopy(self._model.state_dict())
        epochs_without_improvement = 0

        for epoch in range(self._epochs):
            train_loss = self._train_one_epoch(optimizer, loss_fn, interactions)
            val_loss = (
                self._compute_loss(loss_fn, val_tensors) if val_tensors is not None else train_loss
            )
            if epoch_callback is not None:
                epoch_callback(epoch, train_loss, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self._model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self._patience:
                    break

        self._model.load_state_dict(best_state)
        return self

    def predict(self, user_ids: pd.Series, item_ids: pd.Series) -> pd.Series:
        user_idx = torch.tensor(user_ids.to_numpy(), dtype=torch.long)
        item_idx = torch.tensor(item_ids.to_numpy(), dtype=torch.long)
        self._model.eval()
        chunk = self._batch_size * 8
        scores: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(user_idx), chunk):
                end = start + chunk
                batch_scores = self._model(user_idx[start:end], item_idx[start:end])
                scores.append(batch_scores.numpy())
        return pd.Series(np.concatenate(scores) if scores else [], index=user_ids.index)

    @property
    def state_dict(self) -> dict:
        return self._model.state_dict()

    def load_state(self, state_dict: dict) -> "MLPRecommender":
        """Carrega pesos previamente treinados na rede."""
        self._model.load_state_dict(state_dict)
        return self

    def _train_one_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        interactions: pd.DataFrame,
    ) -> float:
        user_idx, item_idx, label = self._build_tensors(interactions)
        permutation = torch.randperm(len(label))
        self._model.train()

        total_loss, n_batches = 0.0, 0
        for start in range(0, len(label), self._batch_size):
            batch = permutation[start : start + self._batch_size]
            optimizer.zero_grad()
            predictions = self._model(user_idx[batch], item_idx[batch])
            loss = loss_fn(predictions, label[batch])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / n_batches

    def _compute_loss(
        self, loss_fn: nn.Module, tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> float:
        user_idx, item_idx, label = tensors
        self._model.eval()
        with torch.no_grad():
            predictions = self._model(user_idx, item_idx)
            return loss_fn(predictions, label).item()

    def _build_tensors(
        self, interactions: pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positives = interactions[["user_idx", "item_idx"]].copy()
        positives["label"] = 1.0
        negatives = self._sample_negatives(positives)
        combined = pd.concat([positives, negatives], ignore_index=True)
        user_idx = torch.tensor(combined["user_idx"].to_numpy(), dtype=torch.long)
        item_idx = torch.tensor(combined["item_idx"].to_numpy(), dtype=torch.long)
        label = torch.tensor(combined["label"].to_numpy(), dtype=torch.float32)
        return user_idx, item_idx, label

    def _sample_negatives(self, positives: pd.DataFrame) -> pd.DataFrame:
        negatives = positives.loc[positives.index.repeat(self._negative_ratio)].copy()
        negatives["item_idx"] = self._rng.integers(0, self._n_items, size=len(negatives))
        negatives["label"] = 0.0
        return negatives
