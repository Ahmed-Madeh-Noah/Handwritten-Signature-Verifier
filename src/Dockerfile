FROM python:3.12

RUN apt-get update
RUN apt-get install -y libgl1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /src

COPY pyproject.toml pyproject.toml

RUN ["uv", "sync"]

COPY . .
