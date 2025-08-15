from __future__ import annotations

from typing import Optional, Tuple
import os
import numpy as np


class AnnIndex:
    def __init__(self, dim: int, engine: str = "auto") -> None:
        self.dim = dim
        self.engine = engine
        self.faiss_index = None
        self.hnsw_index = None

        if engine in ("auto", "faiss"):
            try:
                import faiss  # type: ignore
                self.faiss = faiss
            except Exception:
                self.faiss = None
        else:
            self.faiss = None

        if engine in ("auto", "hnsw"):
            try:
                import hnswlib  # type: ignore
                self.hnswlib = hnswlib
            except Exception:
                self.hnswlib = None
        else:
            self.hnswlib = None

    def build(self, vectors: np.ndarray, ef_construction: int = 200, M: int = 16) -> None:
        if self.faiss is not None:
            self.faiss_index = self.faiss.IndexFlatIP(self.dim)
            self.faiss_index.add(vectors.astype("float32"))
            return
        if self.hnswlib is not None:
            self.hnsw_index = self.hnswlib.Index(space='cosine', dim=self.dim)
            self.hnsw_index.init_index(max_elements=vectors.shape[0], ef_construction=ef_construction, M=M)
            self.hnsw_index.add_items(vectors.astype("float32"))
            self.hnsw_index.set_ef(64)
            return
        # Else, no-op; we will rely on brute-force fallback where used

    def search(self, queries: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if self.faiss_index is not None:
            sims, idxs = self.faiss_index.search(queries.astype("float32"), top_k)
            return idxs, sims
        if self.hnsw_index is not None:
            idxs, sims = self.hnsw_index.knn_query(queries.astype("float32"), k=top_k)
            return idxs, 1.0 - sims  # hnswlib returns distances
        # Brute force cosine sim
        items = self._get_items_matrix()
        if items is None or items.shape[0] == 0:
            return np.zeros((queries.shape[0], 0), dtype=int), np.zeros((queries.shape[0], 0), dtype=float)
        queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9)
        items_norm = items / (np.linalg.norm(items, axis=1, keepdims=True) + 1e-9)
        sims = queries_norm @ items_norm.T
        idxs = np.argsort(-sims, axis=1)[:, :top_k]
        sorted_sims = np.take_along_axis(sims, idxs, axis=1)
        return idxs, sorted_sims

    def _get_items_matrix(self) -> Optional[np.ndarray]:
        # Placeholder - this wrapper is used by higher-level code which can override this
        return None