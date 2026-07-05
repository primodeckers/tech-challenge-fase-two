# Tech Challenge — Fase 02

Sistema de recomendação de produtos para e-commerce baseado no comportamento de navegação dos usuários, com rede neural em PyTorch, pipeline reprodutível com DVC, experimentos rastreados no MLflow e ambiente containerizado com Docker.

> Projeto da Pós-Tech (FIAP) — Machine Learning Engineering, Fase 02.

## Status

🚧 Em desenvolvimento — Etapa 1 concluída (estrutura, clean code, ambiente reprodutível). Próxima: Etapa 2 (Docker).

## Stack

- **PyTorch** — rede neural (MLP / embedding-based, estilo NCF) para recomendação
- **Scikit-Learn** — baselines
- **MLflow** — tracking de experimentos e Model Registry
- **DVC** — versionamento de dados e pipeline reprodutível
- **Docker** — containerização (multi-stage)
- **uv** — gerenciamento de dependências e ambiente Python

## Estrutura do projeto

```
src/
  config/settings.py      # configuração via Pydantic Settings (.env)
  data/                    # load_data() e preprocess_data()
  features/                # encoding de IDs user/item
  models/                  # BaseRecommender (contrato), PopularityRecommender
                            # (baseline), MLPRecommender (rede neural)
  models/factory.py        # ModelFactory — padrão de projeto Factory Method
  training/trainer.py      # ModelTrainer — orquestra o treino
  evaluation/evaluator.py  # RankingEvaluator — precision/recall/ndcg/hit_rate @k
scripts/validate_env.py    # valida configuração antes de rodar o pipeline
tests/                     # testes unitários (pytest)
```

## Como executar

```bash
uv sync                  # instala dependências e cria o ambiente virtual
uv run validar-env       # valida configuração (.env)
uv run pytest            # roda os testes
uv run treinar           # treina o modelo (requer dataset em data/raw/, ver Etapa 3)
```

Copie `.env.example` para `.env` e ajuste conforme necessário.
