# Global Book Recommender (AI/ML/DL/DSA)

An end-to-end, production-oriented book recommendation system that provides global, personalized suggestions using a hybrid of collaborative filtering, content-based ML, deep NLP embeddings, and optimized data structures (FAISS/HNSW for ANN, Neo4j for graph reasoning, Redis for caching).

## Features
- Hybrid recommender (collaborative + content + deep embeddings)
- Multilingual, global catalog (country/language-aware)
- Real-time similarity search via ANN indices (FAISS/HNSW) with graceful fallbacks
- Graph-based reasoning via Neo4j (optional)
- Explanations for each recommendation
- Scalable API with FastAPI; deployable via Docker/Kubernetes; model serving via TF Serving/ONNX (optional)

## Quickstart (Core, CPU-only)
1) Create a virtual environment and install core dependencies:

```bash
make create-venv
make install-core
```

2) Run unit tests:
```bash
make test
```

3) Start the API:
```bash
make run-api
```

Open: http://127.0.0.1:8000/docs

4) Try recommendations (example):
- POST /recommend with JSON body:
```json
{
  "genres": ["Fantasy"],
  "authors": ["Nnedi Okorafor"],
  "countries": ["Nigeria"],
  "languages": ["en"],
  "themes": ["Afrofuturism"],
  "min_year": 2010,
  "max_year": 2025,
  "liked_books": ["Akata Witch"],
  "limit": 5
}
```

## Enable Advanced ML/DL (Optional)
- Install ML extras (FAISS/HNSW/Transformers/Torch/Neo4j/Redis):
```bash
make install-ml
```
- Build ANN index from data (optional, falls back if unavailable):
```bash
make build-index
```

## Repo Structure
- `services/api/` — FastAPI app and routers
- `recommender/` — hybrid/content/collaborative models and retrieval
- `search/` — FAISS/HNSW wrappers (optional)
- `graph/` — Neo4j client and path-based recommendations (optional)
- `storage/` — caching (Redis) abstractions (optional)
- `data_pipeline/` — ingestion, preprocessing, schemas
- `evaluation/` — metrics (Precision@k, Recall@k, NDCG)
- `scripts/` — utilities to ingest/build indices
- `docs/` — architecture, pseudocode, deployment
- `ui_mockups/` — static mock UI files for filters/recommendations
- `sample_data/` — small global dataset for demos/tests

## Documentation
- Architecture: `docs/architecture.md`
- Pseudocode: `docs/pseudocode.md`
- Deployment: `docs/deployment.md`

## Ethics & Explainability
- Bias mitigation via balanced sampling, diversity controls, and region/language-aware ranking
- Explanation strings attached to each recommendation (e.g., "Matches your preference for dystopian themes")

## License
MIT