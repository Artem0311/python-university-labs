FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN uv sync --locked --no-dev

FROM python:3.13-slim-bookworm

WORKDIR /app

RUN useradd --system --create-home --uid 10001 findex

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

COPY index.bin /app/index.bin
COPY data/test.txt /app/data/test.txt
COPY data/wiki/*.txt /app/data/wiki/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    INDEX_PATH="/app/index.bin" \
    CORPUS_ROOT="/app/data"

USER findex

EXPOSE 8000

CMD ["findex", "serve", "--port", "8000"]
