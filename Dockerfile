FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1 PORT=8000 FILES_PATH=/files VECTOR_DB_PATH=/app/data/chroma_db

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
