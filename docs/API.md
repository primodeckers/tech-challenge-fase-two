# API de Inferência — Documentação de Uso

API HTTP (FastAPI) que serve o modelo de recomendação treinado. Dado um usuário,
retorna os produtos com maior afinidade prevista.

- **URL pública (produção):** https://recomendador-api-447z.onrender.com
- **Local:** `http://localhost:8000` (ver seção *Como rodar localmente*)

> A API não recebe corpo JSON: os parâmetros vão na própria URL (endpoints GET).

## Endpoints

| Método | Rota | Parâmetros | Descrição |
|---|---|---|---|
| GET | `/` | — | Apresentação do serviço e lista de endpoints |
| GET | `/health` | — | Status do serviço e se o modelo está carregado |
| GET | `/recomendar/{user_idx}` | `user_idx` (path, obrigatório), `k` (query, opcional, padrão 10) | Top-k itens recomendados para o usuário |
| GET | `/docs` | — | Documentação interativa (Swagger UI, gerada pelo FastAPI) |
| GET | `/openapi.json` | — | Esquema OpenAPI da API |

- `user_idx`: índice do usuário (inteiro de `0` a `75847`).
- `k`: número de recomendações desejado (inteiro).

## Exemplos (curl)

```bash
# status
curl https://recomendador-api-447z.onrender.com/health
# -> {"status":"ok","model_loaded":true}

# top-5 recomendações para o usuário 0
curl "https://recomendador-api-447z.onrender.com/recomendar/0?k=5"
```

A resposta do `/recomendar` tem este formato (os valores abaixo são só ilustrativos —
os itens e scores reais dependem do modelo em produção):

```json
{
  "user_idx": 0,
  "recomendacoes": [
    { "item_idx": 3316, "score": 0.9738 },
    { "item_idx": 1421, "score": 0.9644 }
  ]
}
```

`item_idx` é o índice interno do produto e `score` é a afinidade prevista (0 a 1).

## Testar pelo Swagger (navegador)

1. Abra **https://recomendador-api-447z.onrender.com/docs**
2. Clique no endpoint **`GET /recomendar/{user_idx}`** para expandir.
3. Clique em **Try it out**.
4. Preencha os campos (são apenas números, **não há corpo JSON**):
   - **user_idx**: ex. `0`
   - **k**: ex. `5`
5. Clique em **Execute**. A resposta aparece em **Response body** (código `200`).

## Como rodar localmente

```bash
# via uvicorn
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# ou via container
docker build -f Dockerfile.api -t recomendador-api .
docker run -p 8000:8000 recomendador-api
```

Depois acesse `http://localhost:8000/docs`.

## Observações

- O deploy usa o free tier do Render, que **hiberna após inatividade**: a primeira
  requisição depois de um tempo ocioso pode levar ~50s (cold start). As seguintes são
  rápidas.
- O modelo servido é embutido na imagem (`deploy/artifacts/model.pt`); a dimensão do
  embedding é inferida automaticamente do próprio arquivo de pesos.
