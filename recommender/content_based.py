from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_pipeline.schemas import RecommendationRequest


class ContentBasedRecommender:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix: np.ndarray | None = None
        self.book_ids: list[str] = []
        self.books_df: pd.DataFrame | None = None

    def fit(self, books_df: pd.DataFrame) -> None:
        self.books_df = books_df.copy()
        self.book_ids = [str(x) for x in self.books_df["book_id"].tolist()]
        corpus = self._build_corpus(self.books_df)
        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def _build_corpus(self, df: pd.DataFrame) -> list[str]:
        corpus: list[str] = []
        for _, row in df.iterrows():
            fields = [
                str(row.get("title", "")),
                str(row.get("author", "")),
                str(row.get("genres", "")).replace("|", " "),
                str(row.get("themes", "")).replace("|", " "),
                str(row.get("description", "")),
                str(row.get("country", "")),
                str(row.get("language", "")),
            ]
            corpus.append(" ".join([t for t in fields if t]))
        return corpus

    def _request_to_query_text(self, request: RecommendationRequest) -> str:
        tokens = []
        tokens.extend(request.genres)
        tokens.extend(request.themes)
        tokens.extend(request.authors)
        tokens.extend(request.countries)
        tokens.extend(request.languages)
        tokens.extend(request.liked_books)
        return " ".join(tokens)

    def score_candidates(self, request: RecommendationRequest, candidate_df: pd.DataFrame) -> Dict[str, float]:
        if self.vectorizer is None or self.tfidf_matrix is None or self.books_df is None:
            return {}
        if candidate_df.empty:
            return {}
        query = self._request_to_query_text(request)
        if not query.strip():
            # No preferences -> use popularity proxy (rating_count * avg_rating)
            scores: Dict[str, float] = {}
            for _, row in candidate_df.iterrows():
                rc = float(row.get("rating_count", 0.0))
                ar = float(row.get("avg_rating", 0.0))
                scores[str(row["book_id"])] = rc * ar
            return scores
        query_vec = self.vectorizer.transform([query])

        # Map candidate indices to global tfidf matrix rows
        id_to_idx = {bid: i for i, bid in enumerate(self.book_ids)}
        candidate_indices = [id_to_idx.get(str(bid), -1) for bid in candidate_df["book_id"].astype(str).tolist()]
        valid_mask = [i for i, idx in enumerate(candidate_indices) if idx >= 0]
        if not valid_mask:
            return {}
        idxs = [candidate_indices[i] for i in valid_mask]
        sub_matrix = self.tfidf_matrix[idxs]
        sims = cosine_similarity(query_vec, sub_matrix)[0]
        scores: Dict[str, float] = {}
        candidate_ids = candidate_df["book_id"].astype(str).tolist()
        for local_i, sim in zip(valid_mask, sims):
            bid = str(candidate_ids[local_i])
            scores[bid] = float(sim)
        return scores