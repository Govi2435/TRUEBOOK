### Architecture Overview

- **Data Sources**: Open Library, Google Books, Goodreads (optional), user interactions
- **Ingestion** (`data_pipeline/`): Fetch, normalize, and store to `sample_data/` (demo) or data lake (prod)
- **Preprocessing**: NLP cleaning, feature encoding, TF-IDF matrix, optional BERT embeddings
- **Models** (`recommender/`):
  - Content-based (TF-IDF, optional BERT)
  - Collaborative (co-occurrence/popularity, optional matrix factorization)
  - Hybrid fusion with diversity control
- **ANN Index** (`search/`): FAISS/HNSW, with brute-force fallback
- **Graph** (`graph/`): Neo4j for path-based discovery (optional)
- **API** (`services/api/`): FastAPI endpoints `/health`, `/recommend`
- **Cache** (`storage/`): Redis for hot queries
- **Evaluation** (`evaluation/`): Precision@k, Recall@k, NDCG, diversity

### Data Flow
1. Ingest raw books + interactions
2. Preprocess and persist features
3. Train/fit recommenders and build indices
4. Serve via API; query-time fusion and explanation

### Diagram (text)

[User] -> [FastAPI] -> [Hybrid Recommender]
  |                         |--(content)--> [TF-IDF/BERT]
  |                         |--(collab)--> [Co-occurrence/SVD]
  |                         |--(ANN)-----> [FAISS/HNSW]
  |                         |--(graph)--> [Neo4j]
  |                         |--(cache)--> [Redis]
  \-> [Storage/Logs/Metrics]