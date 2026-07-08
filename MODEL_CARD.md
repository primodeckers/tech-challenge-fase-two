# Model Card — Recomendador de Produtos (RetailRocket)

Documentação do modelo seguindo a estrutura de *Model Cards* (Mitchell et al., 2019).

## Detalhes do modelo

- **Nome:** `RecomendadorEcommerce` (MLflow Model Registry)
- **Versão em produção:** v3 (alias `@production`)
- **Tipo:** recomendação com feedback implícito (filtragem colaborativa neural)
- **Arquitetura:** *embedding-based MLP* no estilo Neural Collaborative Filtering
  — embeddings de usuário e item (dim. 64) concatenados → MLP (64 → ReLU →
  Dropout 0.2 → 1) → sigmoide (probabilidade de interação).
- **Framework:** PyTorch 2.x (CPU)
- **Treino:** mini-batches, amostragem negativa (1:1) reamostrada por época,
  otimizador Adam, perda BCE, *early stopping* pela perda de validação com
  restauração do melhor estado. Semente fixa (42) para reprodutibilidade.

## Uso pretendido

- **Objetivo:** dado um usuário, estimar a afinidade com itens do catálogo para
  ranquear recomendações de produtos em e-commerce.
- **Usuários-alvo:** times de personalização/recomendação.
- **Fora de escopo:** o modelo não usa atributos de conteúdo (preço, categoria,
  texto) nem contexto de sessão em tempo real; não deve ser usado como única
  base para decisões sensíveis.

## Dados

- **Fonte:** [RetailRocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
  (eventos `view` / `addtocart` / `transaction`, ~4,5 meses).
- **Pré-processamento:** eventos positivos como sinal implícito; remoção de
  duplicatas; filtro de usuários e itens com ≥ 5 interações.
- **Após limpeza:** ~878k interações, 75.848 usuários, 65.706 itens.
- **Split:** *leave-one-out* por usuário — interação mais recente para teste,
  segunda mais recente para validação, restante para treino.
- **Versionamento:** dados e modelo versionados com DVC; cada modelo é rastreável
  até o commit de código e a versão dos dados que o geraram.

## Avaliação

- **Protocolo:** para cada usuário, o item retido é ranqueado contra 99 itens
  negativos amostrados (protocolo NCF). Métricas com corte @10.

| Métrica | Modelo em produção (emb=64) | Baseline (popularidade) |
|---|---|---|
| HitRate@10 | **0.486** | 0.447 |
| NDCG@10 | **0.278** | 0.259 |
| Precision@10 | **0.049** | 0.045 |
| Recall@10 | **0.486** | 0.447 |

A rede neural supera o baseline de popularidade em todas as métricas. A seleção
entre as configurações testadas (embeddings 16/32/64) foi feita por NDCG@10 no
conjunto de teste — a versão com embeddings de dimensão 64 foi a melhor e a
promovida a produção.

## Hiperparâmetros (modelo em produção)

| Parâmetro | Valor |
|---|---|
| embedding_dim | 64 |
| learning_rate | 0.001 |
| epochs (máx.) | 20 |
| patience (early stopping) | 3 |
| negative_ratio (treino) | 1 |
| seed | 42 |

## Limitações e vieses

- **Viés de popularidade:** treinado com feedback implícito (predominância de
  `view`), tende a favorecer itens populares — mitigado em parte pela amostragem
  negativa, mas presente.
- **Cold start:** não recomenda para usuários/itens ausentes do treino (sem
  embedding aprendido); exige retreino para incorporar novos IDs.
- **Sem features de conteúdo:** ignora preço, categoria e texto — recomendações
  puramente colaborativas.
- **Drift:** o comportamento de compra muda ao longo do tempo; requer
  monitoramento e retreino periódico.

## Considerações éticas

- Dados anonimizados (IDs numéricos), sem informação pessoal identificável.
- Recomendações devem ser complementadas por regras de negócio (ex.: evitar
  reforçar bolhas de consumo ou expor itens inadequados).

## Reprodução

```bash
uv sync
dvc repro           # pipeline canônico: preprocess → featurize → train → evaluate
uv run experimento  # 3 runs comparados no MLflow + promoção da melhor a produção
```

O modelo em produção pode ser carregado via `models:/RecomendadorEcommerce@production`.
