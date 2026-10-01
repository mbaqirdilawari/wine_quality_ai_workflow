FROM python:3.13-slim

WORKDIR /app

# Install dependencies first so this layer is cached between code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY wine_analysis/ wine_analysis/
COPY tests/ tests/
COPY data/ data/
COPY setup.cfg .

ENV WINE_DATA_PATH=/app/data/wine_quality_merged.csv \
    WINE_OUTPUT_DIR=/app/outputs \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "wine_analysis.main"]
