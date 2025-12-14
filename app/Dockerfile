FROM python:3.12

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY . .

RUN ["uv", "sync"]
