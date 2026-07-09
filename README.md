# Tech Challenge — Fase 02

Sistema de recomendação de produtos para e-commerce baseado no comportamento de navegação dos usuários, com rede neural em PyTorch, pipeline reprodutível com DVC, experimentos rastreados no MLflow e ambiente containerizado com Docker.

> Projeto da Pós-Tech (FIAP) — Machine Learning Engineering, Fase 02.

## Status

🚧 Em finalização — Etapas 1–4 completas: estrutura + clean code; Docker; pipeline DVC
de 4 estágios + rede neural avaliada; MLflow com múltiplos runs, Model Registry (melhor
modelo em produção) e [Model Card](MODEL_CARD.md); **API de inferência publicada na
nuvem** (bônus). Pendente: vídeo.

## API em produção (deploy na nuvem) 🌐

A API de inferência está publicada em um web service Docker no Render, acessível por
URL pública:

**https://recomendador-api-447z.onrender.com**

| Endpoint | Descrição |
|---|---|
| [`/health`](https://recomendador-api-447z.onrender.com/health) | status do serviço e do modelo |
| [`/recomendar/0?k=5`](https://recomendador-api-447z.onrender.com/recomendar/0?k=5) | top-k itens recomendados para um usuário |
| [`/docs`](https://recomendador-api-447z.onrender.com/docs) | documentação interativa (Swagger) |

O deploy é automático a partir do `render.yaml` (blueprint) usando a imagem enxuta
`Dockerfile.api`. Rodar localmente: `uvicorn src.api.app:app` ou
`docker build -f Dockerfile.api -t recomendador-api . && docker run -p 8000:8000 recomendador-api`.

> O free tier hiberna após inatividade — a primeira requisição pode levar ~50s (cold start).

## Dataset

[**RetailRocket**](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) — eventos de e-commerce (`view` / `addtocart` / `transaction`). Após a limpeza, ~878k interações (75,8k usuários, 65,7k itens). O `events.csv` é versionado com DVC (não vai para o Git); baixe-o do Kaggle e coloque em `data/raw/events.csv`.

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
  config/                  # Pydantic Settings (.env) e loader de params (params.yaml)
  data/                    # load_data() e preprocess_data()
  features/                # encoding de IDs e split leave-one-out
  models/                  # BaseRecommender (contrato), PopularityRecommender
                            # (baseline), MLPRecommender (rede neural)
  models/factory.py        # ModelFactory — padrão de projeto Factory Method
  training/trainer.py      # ModelTrainer — orquestra o treino
  evaluation/evaluator.py  # RankingEvaluator — precision/recall/ndcg/hit_rate @k
  pipeline/                # estágios do DVC: preprocess, featurize, train, evaluate
configs/params.yaml        # hiperparâmetros (fonte única de config)
scripts/validate_env.py    # valida configuração antes de rodar o pipeline
tests/                     # testes unitários (pytest)
```

## Pipeline DVC

O pipeline (`dvc.yaml`) tem quatro estágios encadeados, reexecutados de forma
incremental (só o que muda) com `dvc repro`:

| Estágio | Entrada → Saída | O que faz |
|---|---|---|
| `preprocess` | `events.csv` → `interactions.parquet` | limpa e filtra interações |
| `featurize` | `interactions` → `train/val/test.parquet` | codifica IDs e split leave-one-out |
| `train` | splits → `model.pt` | treina a rede neural (log no MLflow) |
| `evaluate` | `model.pt` + teste → `metrics.json` | métricas de ranking @k |

```bash
dvc repro                 # executa o pipeline completo (ou só o que mudou)
dvc metrics show          # mostra as métricas do último run
```

### Armazenamento remoto (S3)

Dados e modelos são versionados no DVC e sincronizados com um bucket **Amazon S3**
(`s3://tech-challenge-dvc-primodeckers/tech-challenge`, região `us-east-2`):

```bash
dvc push                  # envia dados/modelos versionados para o S3
dvc pull                  # baixa a versão correspondente ao commit atual
```

As credenciais AWS ficam em `~/.aws/credentials` (fora do repositório); o
`.dvc/config` versiona apenas a URL do bucket e a região — nenhum segredo.

### Resultados (@10, protocolo NCF: 1 positivo vs 99 negativos)

| Métrica | MLP (rede neural) | Baseline (popularidade) |
|---|---|---|
| HitRate@10 | **0.479** | 0.447 |
| NDCG@10 | **0.274** | 0.259 |
| Precision@10 | **0.048** | 0.045 |
| Recall@10 | **0.479** | 0.447 |

A rede neural supera o baseline em todas as métricas.

## Experimentos e Model Registry (MLflow)

O pipeline registra cada treino no MLflow (parâmetros, curva de perda e métricas
de teste no mesmo run). Além do run canônico do `dvc repro`, há um runner de
experimentos que compara múltiplas configurações e promove a melhor a produção:

```bash
uv run experimento        # treina embeddings 16/32/64, compara e promove a melhor
mlflow ui --backend-store-uri sqlite:///mlflow.db   # UI em http://localhost:5000
```

- **≥3 runs comparáveis** no experimento `recomendador-ecommerce`.
- **Model Registry:** cada configuração vira uma versão de `RecomendadorEcommerce`;
  a melhor (por NDCG@10 — embeddings 64) recebe o alias **`@production`**.
- Carregar o modelo de produção: `models:/RecomendadorEcommerce@production`.

Veja o [Model Card](MODEL_CARD.md) para detalhes do modelo, dados, resultados e limitações.

## Como executar

```bash
uv sync                  # instala dependências e cria o ambiente virtual
uv run validar-env       # valida configuração (.env)
uv run pytest            # roda os testes
dvc repro                # executa o pipeline completo (preprocess → evaluate)
```

Copie `.env.example` para `.env` e ajuste conforme necessário.

## Docker

A imagem é construída em **multi-stage** (estágio de build resolve dependências com uv; estágio de runtime é enxuto e roda como usuário não-root `mluser`). O `docker-compose.yml` orquestra dois serviços:

- **`mlflow`** — servidor de tracking e Model Registry (UI em http://localhost:5000)
- **`train`** — executa o pipeline de treino, registrando no servidor MLflow

```bash
docker compose up mlflow          # sobe só o servidor MLflow (UI em :5000)
docker compose up --build train   # constrói e roda o treino (requer dataset, ver Etapa 3)
docker compose down               # encerra os serviços
```

> A imagem usa `torch` CPU-only (o projeto não requer GPU), o que a mantém enxuta.
