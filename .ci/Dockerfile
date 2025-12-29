FROM registry.int.e-kama.com/devops1/shared/public-images/python:3.12.2-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ENV RYE_HOME="/opt/rye"
ENV PATH="$RYE_HOME/shims:$PATH"
ENV RYE_VERSION="0.39.0"

RUN curl -sSf https://rye.astral.sh/get | RYE_NO_AUTO_INSTALL=1 RYE_INSTALL_OPTION="--yes" bash

COPY requirements.lock requirements-dev.lock pyproject.toml ./

RUN pip install --no-cache-dir -r requirements.lock -r requirements-dev.lock

COPY . /app
