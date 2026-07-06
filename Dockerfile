# syntax=docker/dockerfile:1

# ---------- Estágio de build: resolve e instala dependências ----------
FROM python:3.12-slim AS build

# uv oficial, copiado como binário estático (sem instalar via pip)
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Instala só as dependências primeiro (camada estável, aproveita cache)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copia o código e instala o próprio projeto
COPY src ./src
COPY scripts ./scripts
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------- Estágio de runtime: imagem enxuta, usuário não-root ----------
FROM python:3.12-slim AS runtime

# Usuário sem privilégios (hardening)
RUN groupadd --system mluser && useradd --system --gid mluser --create-home mluser

WORKDIR /app

# Copia o ambiente virtual já resolvido e o código
COPY --from=build --chown=mluser:mluser /app/.venv /app/.venv
COPY --chown=mluser:mluser src ./src
COPY --chown=mluser:mluser scripts ./scripts
COPY --chown=mluser:mluser configs ./configs

# Ativa o venv e evita buffering de logs
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER mluser

# Por padrão, executa o pipeline de treino
CMD ["treinar"]
