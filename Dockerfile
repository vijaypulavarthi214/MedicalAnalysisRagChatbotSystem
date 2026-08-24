FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .

# CPU-only torch keeps the image well under Render's build limits — the GPU
# wheel is several GB larger and unused in this deployment.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Bake the embedding model into the image at build time so the first request
# on Render doesn't pay a slow Hugging Face download during a cold start.
# Trade-off: a larger image (~1-1.5GB with torch + model weights) for faster,
# more predictable cold starts.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY backend/app ./app

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
