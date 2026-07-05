import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models.base import BaseRecommender


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
    """Recomendador neural com embeddings de usuário/item, treino via negative sampling."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 32,
        learning_rate: float = 1e-3,
        epochs: int = 20,
        patience: int = 3,
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        self._model = _EmbeddingMLP(n_users, n_items, embedding_dim)
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._patience = patience
        self._rng = np.random.default_rng(seed)

    def fit(self, interactions: pd.DataFrame) -> "MLPRecommender":
        """Treina com amostragem negativa e early stopping em cima da loss de treino.

        Espera colunas 'user_idx' e 'item_idx' (índices contíguos já codificados).
        """
        user_idx, item_idx, label = self._build_training_tensors(interactions)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
        loss_fn = nn.BCELoss()

        best_loss = float("inf")
        epochs_without_improvement = 0
        self._model.train()
        for _ in range(self._epochs):
            loss = self._train_one_epoch(optimizer, loss_fn, user_idx, item_idx, label)
            if loss < best_loss:
                best_loss, epochs_without_improvement = loss, 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self._patience:
                break
        return self

    def predict(self, user_ids: pd.Series, item_ids: pd.Series) -> pd.Series:
        user_idx = torch.tensor(user_ids.to_numpy(), dtype=torch.long)
        item_idx = torch.tensor(item_ids.to_numpy(), dtype=torch.long)
        self._model.eval()
        with torch.no_grad():
            scores = self._model(user_idx, item_idx)
        return pd.Series(scores.numpy(), index=user_ids.index)

    def _train_one_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        label: torch.Tensor,
    ) -> float:
        optimizer.zero_grad()
        predictions = self._model(user_idx, item_idx)
        loss = loss_fn(predictions, label)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _build_training_tensors(
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
        n_items = self._model.item_embedding.num_embeddings
        negatives = positives.copy()
        negatives["item_idx"] = self._rng.integers(0, n_items, size=len(positives))
        negatives["label"] = 0.0
        return negatives
