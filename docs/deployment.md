### Deployment Strategy

- **API**: FastAPI on Uvicorn/Gunicorn; containerize with Docker
- **Models**: Embed TF-IDF in app; serve heavy DL via TF Serving/ONNXRuntime if needed
- **ANN**: Build FAISS/HNSW index offline; mount read-only volume to API pods
- **Cache**: Redis managed service
- **Graph**: Neo4j Aura or self-hosted
- **Scaling**: Horizontal autoscaling on K8s; shard by region/language for data locality

### Example Dockerfile (API)
```
FROM python:3.11-slim
WORKDIR /app
COPY requirements-core.txt ./
RUN pip install --no-cache-dir -r requirements-core.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Observability
- Traces: OpenTelemetry
- Metrics: request latency, rec latency, cache hit rate, Precision@k
- Logs: structured JSON; redact PII