FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements-runtime.txt .

RUN python -m pip install --no-cache-dir -r requirements-runtime.txt \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu"

COPY . .

RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8001 8002 8003

CMD ["python", "-m", "uvicorn", "services.coordinator_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
