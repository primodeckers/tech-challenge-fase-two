---
title: Recomendador Ecommerce
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# Recomendador de Produtos — API de Inferência

Deploy da API FastAPI do Tech Challenge Fase 02 (sistema de recomendação de
e-commerce) em um Hugging Face Space com SDK Docker.

Serve o modelo neural treinado (embeddings estilo NCF) via HTTP:

- `GET /health` — status do serviço e do modelo.
- `GET /recomendar/{user_idx}?k=10` — top-k itens recomendados para o usuário.
- `GET /docs` — documentação interativa (Swagger).

Repositório do projeto: https://github.com/primodeckers/tech-challenge-fase-two
