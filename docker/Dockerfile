# Dockerfile (multi-stage)
FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY app/ ./app/
CMD ["python", "app/main.py"]