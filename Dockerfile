# Demo-only image: the review UI serves a precomputed batch, so it needs neither
# torch nor the ML stack. Keeping them out holds the image to a few hundred MB and
# makes the free tier viable.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    UNDERWRITE_BATCH=/app/reports/demo_batch.json

WORKDIR /app

RUN pip install --no-cache-dir "fastapi>=0.111" "uvicorn>=0.30"

COPY src ./src
COPY reports/demo_batch.json ./reports/demo_batch.json

EXPOSE 8000

CMD ["uvicorn", "underwrite.app.server:app", "--host", "0.0.0.0", "--port", "8000"]
