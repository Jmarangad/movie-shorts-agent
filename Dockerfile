# syntax=docker/dockerfile:1
# ---------------------------------------------------------------
# Stage 1: python base with OS runtime deps (ffmpeg, imagemagick,
# Deva fonts for captions, unzip for nohup + build tools cleanup).
# ---------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ImageMagick policy: allow @ (file reads) and raise resource limits so
# TextClip rendering cannot be killed. Deva fonts: Devanagari captions.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        imagemagick \
        fonts-deva \
        fonts-indic \
        curl \
    && sed -i \
        -e 's/rights="none" pattern="\*"/rights="read|write" pattern="\*\"/' \
        -e 's/<policy domain="resource" name="memory" value=".*"/<policy domain="resource" name="memory" value="2GiB"\/>/' \
        -e 's/<policy domain="resource" name="map" value=".*"/<policy domain="resource" name="map" value="4GiB"\/>/' \
        -e 's/<policy domain="resource" name="width" value=".*"/<policy domain="resource" name="width" value="32KP"\/>/' \
        -e 's/<policy domain="resource" name="height" value=".*"/<policy domain="resource" name="height" value="32KP"\/>/' \
        /etc/ImageMagick-*/policy.xml \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------
# Stage 2: install python deps into a venv (cache-friendly).
# ---------------------------------------------------------------
FROM base AS deps

COPY requirements.txt .
RUN python -m venv /venv \
    && /venv/bin/pip install --upgrade pip \
    && /venv/bin/pip install -r requirements.txt

# ---------------------------------------------------------------
# Stage 3: final runtime, non-root, code only.
# ---------------------------------------------------------------
FROM base AS runtime

COPY --from=deps /venv /venv
ENV PATH="/venv/bin:$PATH"

RUN groupadd -r agent && useradd -r -g agent -u 1000 -m -d /home/agent agent \
    && mkdir -p /app/output /app/downloads \
    && chown -R agent:agent /app

COPY --chown=agent:agent config.py main.py ./
COPY --chown=agent:agent src/ ./src/

USER agent
ENV OUTPUT_DIR=/app/output \
    DOWNLOADS_DIR=/app/downloads

ENTRYPOINT ["/venv/bin/python", "main.py"]