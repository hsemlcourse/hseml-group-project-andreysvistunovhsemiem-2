# Воспроизводимая среда для обучения моделей.
# uv пересоздаёт окружение из uv.lock 1:1.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv для управления зависимостями
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# Сначала только манифесты — кешируем слой с зависимостями
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Теперь сам код
COPY src/ ./src/
COPY tests/ ./tests/
RUN uv sync --frozen --no-dev

ENV PATH="/opt/venv/bin:${PATH}"

# По умолчанию — обучение пайплайна. Точку входа можно переопределить в compose.
CMD ["python", "-m", "src.modeling"]
