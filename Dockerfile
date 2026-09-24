FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHURN_ROOT=/app

WORKDIR /app

# libgomp : OpenMP requis par LightGBM et XGBoost.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

COPY models/churn_model.joblib models/metadata.json ./models/

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "churn.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
