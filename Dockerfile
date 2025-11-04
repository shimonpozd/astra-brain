FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build tools required for uvicorn[standard], redis[hiredis], etc.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

RUN chmod +x /app/docker-entrypoint.sh

ENV PYTHONPATH="/app${PYTHONPATH:+:${PYTHONPATH}}" \
    ASTRA_CONFIG_ROOT="/app"

EXPOSE 7030

ENTRYPOINT ["./docker-entrypoint.sh"]
