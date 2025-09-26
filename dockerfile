# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm

# ---- security & ergonomics ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XDG_CACHE_HOME=/tmp/.cache \
    TZ=UTC \
    HOME=/home/freelance_user

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata \
  && rm -rf /var/lib/apt/lists/*

# ---- app user (non-root) ----
ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd -g ${APP_GID} freelance_group \
 && useradd -r -u ${APP_UID} -g ${APP_GID} -d /home/freelance_user -m freelance_user

WORKDIR /app

# ---- Python deps first (cache-friendly) ----
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir -r /app/requirements.txt

# ---- App code ----
COPY . /app
RUN chown -R ${APP_UID}:${APP_GID} /app

# ---- Runtime ----
USER ${APP_UID}:${APP_GID}
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
