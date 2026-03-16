# ---- build stage ----
FROM python:3.13-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- runtime stage ----
FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libopus0 \
    libsodium23 \
    && rm -rf /var/lib/apt/lists/*

# copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app
COPY . .

# whisper model cache — mount a volume here to persist across restarts
VOLUME /root/.cache/huggingface

CMD ["python", "bot.py"]
