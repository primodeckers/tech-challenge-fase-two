# Tech Challenge — Fase 02 · Recomendador de e-commerce

Projeto da Pós-Tech FIAP (Machine Learning Engineering, Fase 02). A proposta era
resolver um problema de recomendação de produtos de e-commerce de ponta a ponta —
não só treinar um modelo, mas montar toda a engenharia em volta: código organizado,
ambiente reprodutível, pipeline versionado, experimentos rastreados e o modelo
containerizado e no ar.

O modelo em si é uma rede neural em PyTorch (embeddings de usuário e item, no estilo
Neural Collaborative Filtering). O resto — DVC, MLflow, Docker, uv — é o que faz esse
modelo virar algo reproduzível e operável.

## Entregáveis

- 🎥 **Vídeo (apresentação STAR):** https://youtu.be/RDWVXD1AGZE
- 🌐 **API pública (modelo no ar):** https://recomendador-api-447z.onrender.com
- 📦 **Repositório:** este repositório

## Índice rápido

- [Como rodar](#como-rodar) · [Pipeline (DVC)](#pipeline-de-dados-e-modelo-dvc) ·
  [Experimentos (MLflow)](#experimentos-e-model-registry-mlflow) ·
  [Docker](#docker) · [API na nuvem](#api-em-produção)
- [Model Card](MODEL_CARD.md) · [Uso da API](docs/API.md)

## Dataset

Usamos o [**RetailRocket**](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset),
que traz eventos reais de e-commerce (`view`, `addtocart`, `transaction`) — bem alinhado
com a ideia de "recomendar pelo comportamento de navegação". São ~2,7 milhões de eventos
brutos; depois de filtrar usuários e itens com pouca interação, sobram ~878 mil interações
(75,8 mil usuários e 65,7 mil itens).

O `events.csv` **não vai para o Git** — é versionado com DVC. Para rodar do zero, baixe o
dataset do Kaggle e coloque o arquivo em `data/raw/events.csv` (ou use `dvc pull`, se tiver
acesso ao remote).

## Stack e por que escolhemos cada peça

- **PyTorch** — a rede neural de recomendação (exigência do desafio, e a ferramenta certa
  para o modelo com embeddings).
- **scikit-learn** — apoio a baselines.
- **DVC** — versionar dados e modelo (que não cabem no Git) e amarrar o pipeline em estágios
  reproduzíveis com `dvc repro`.
- **MLflow** — rastrear cada experimento e gerenciar o ciclo de vida do modelo (Registry).
- **Docker** — empacotar o treino e a API de forma isolada e portável.
- **uv** — gerenciador de dependências/ambiente. Escolhemos no lugar do Poetry pela
  velocidade e por ele já cuidar da versão do Python. Também fixamos o `torch` na variante
  **CPU-only**: o projeto não precisa de GPU, e isso derrubou a imagem Docker de ~9 GB para
  algo bem menor.

## Estrutura do projeto

```
src/
  config/       # Settings via Pydantic (.env) + leitura de params.yaml
  data/         # carregamento e limpeza dos eventos
  features/     # encoding de IDs e split leave-one-out
  models/       # contrato BaseRecommender, baseline de popularidade e a rede (MLP)
  models/factory.py   # ModelFactory — cria o modelo a partir de um nome (Factory Method)
  training/     # ModelTrainer — orquestra o treino
  evaluation/   # métricas de ranking (precision/recall/ndcg/hit_rate @k) e protocolo NCF
  pipeline/     # os estágios do DVC: preprocess, featurize, train, evaluate, experiment
  api/          # a API FastAPI que serve o modelo
configs/params.yaml   # todos os hiperparâmetros num lugar só
tests/          # testes unitários (pytest)
```

Sobre organização: cada responsabilidade ficou num módulo curto (carregar dados ≠ treinar
≠ avaliar), com type hints e funções pequenas. O `ModelFactory` foi o design pattern que
mais fez sentido aqui — cria baseline ou rede neural pelo nome, sem espalhar `if/else`.

## Como rodar

```bash
uv sync                 # cria o ambiente e instala tudo a partir do lock
cp .env.example .env    # ajuste se precisar
uv run validar-env      # confere se a configuração está ok
uv run pytest           # roda os testes
dvc repro               # roda o pipeline inteiro (preprocess -> evaluate)
```

## Pipeline de dados e modelo (DVC)

O `dvc.yaml` encadeia quatro estágios. O `dvc repro` só reexecuta o que mudou — se você
mexe só num hiperparâmetro, ele não reprocessa os dados brutos de novo.

| Estágio | Entrada → saída | O que faz |
|---|---|---|
| `preprocess` | `events.csv` → `interactions.parquet` | limpa, filtra eventos e interações raras |
| `featurize` | `interactions` → `train/val/test.parquet` | codifica IDs e faz o split leave-one-out |
| `train` | splits → `model.pt` | treina a rede (com log no MLflow) |
| `evaluate` | `model.pt` + teste → `metrics.json` | calcula as métricas de ranking |

Para o split, adotamos **leave-one-out por usuário** (a interação mais recente vai para
teste, a penúltima para validação, o resto para treino) — é o padrão em recomendação com
feedback implícito e garante que todo usuário avaliado também aparece no treino.

### Armazenamento remoto (S3)

Os dados e o modelo versionados pelo DVC são guardados num bucket **Amazon S3**
(`s3://tech-challenge-dvc-primodeckers/tech-challenge`, `us-east-2`):

```bash
dvc push    # envia para o S3
dvc pull    # traz de volta a versão do commit atual
```

As chaves da AWS ficam só em `~/.aws/credentials` (nunca no repositório) — o `.dvc/config`
guarda apenas a URL do bucket e a região.

### Resultados

Avaliamos com o protocolo do NCF: para cada usuário, o item retido é ranqueado contra 99
itens negativos amostrados, com corte em @10.

| Métrica | Rede neural (MLP) | Baseline (popularidade) |
|---|---|---|
| HitRate@10 | **0.486** | 0.447 |
| NDCG@10 | **0.278** | 0.259 |
| Precision@10 | **0.049** | 0.045 |
| Recall@10 | **0.486** | 0.447 |

A rede ganha do baseline em todas as métricas. Vale a honestidade: a popularidade é um
baseline forte no RetailRocket (o `view` domina e concentra nos itens populares), então a
diferença é consistente, mas não gigante — o que já era esperado.

## Experimentos e Model Registry (MLflow)

Cada treino é registrado no MLflow — parâmetros, curva de perda por época e as métricas de
teste, tudo no mesmo run. Além do run "oficial" do `dvc repro`, tem um runner de
experimentos que compara configurações e promove a melhor:

```bash
uv run experimento     # treina embeddings 16/32/64, compara e promove a melhor
mlflow ui --backend-store-uri sqlite:///mlflow.db     # abre a UI em localhost:5000
```

- Gera 3 runs comparáveis lado a lado na UI.
- Cada configuração vira uma versão do modelo `RecomendadorEcommerce` no Registry.
- A melhor por NDCG@10 (embeddings 64) recebe o alias **`@production`** — que é a mesma
  config que o pipeline treina por padrão e que a API serve.

Detalhe de ambiente: o MLflow usa um backend SQLite local (`sqlite:///mlflow.db`), então o
Registry funciona sem precisar subir servidor. O `docker-compose.yml` sobe um servidor
MLflow de verdade para quem quiser o fluxo completo.

## Docker

A imagem do treino é **multi-stage** (um estágio resolve as dependências com uv, outro é o
runtime enxuto) e roda como usuário não-root. O `docker-compose.yml` sobe dois serviços:

```bash
docker compose up mlflow             # servidor MLflow (UI em :5000)
docker compose up --build train      # constrói e roda o treino
docker compose down
```

Para a **API**, tem um `Dockerfile.api` separado, ainda mais leve (só as dependências de
runtime da API, com o modelo embutido na imagem).

## API em produção

Publicamos a API num web service Docker no **Render** (o deploy é automático a partir do
`render.yaml`). Está no ar em:

**https://recomendador-api-447z.onrender.com**

| Endpoint | Descrição |
|---|---|
| [`/`](https://recomendador-api-447z.onrender.com/) | apresentação e lista de endpoints |
| [`/health`](https://recomendador-api-447z.onrender.com/health) | status do serviço e do modelo |
| [`/recomendar/0?k=5`](https://recomendador-api-447z.onrender.com/recomendar/0?k=5) | top-k recomendações de um usuário |
| [`/docs`](https://recomendador-api-447z.onrender.com/docs) | Swagger (gerado pelo FastAPI) |

O passo a passo de uso (inclusive como testar pelo Swagger) está em [docs/API.md](docs/API.md).

> Uma ressalva honesta: primeiro tentamos o Hugging Face Spaces, mas Docker Spaces passaram
> a exigir assinatura paga, então migramos para o Render (free tier). Como o free tier
> hiberna após um tempo ocioso, a **primeira** requisição pode levar ~50s para "acordar".

## Testes e qualidade

`ruff` cuida do lint e da formatação (com `pre-commit`), e o `pytest` cobre as peças
principais — encoding, split, baseline, rede neural, avaliador e a API. Um workflow de
**GitHub Actions** roda lint + testes a cada push.
