from __future__ import annotations

from typing import Dict, List
import os
import numpy as np
import pandas as pd

from data_pipeline.schemas import RecommendationRequest


class CollaborativeRecommender:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.user_item: pd.DataFrame | None = None
        self.item_popularity: Dict[str, float] = {}
        self.cooccurrence: Dict[str, Dict[str, int]] = {}
        self.items_set: set[str] = set()

    def fit(self, interactions_csv: str, books: pd.DataFrame | None = None) -> None:
        if not os.path.exists(interactions_csv):
            # Build trivial popularity from books
            if books is not None and not books.empty:
                for _, row in books.iterrows():
                    self.item_popularity[str(row["book_id"])] = float(row.get("rating_count", 1.0)) * float(row.get("avg_rating", 0.0))
            return
        df = pd.read_csv(interactions_csv)
        required_cols = {"user_id", "book_id", "event_strength"}
        if not required_cols.issubset(set(df.columns)):
            return
        df["book_id"] = df["book_id"].astype(str)
        self.user_item = df
        # Compute popularity
        pop = df.groupby("book_id")["event_strength"].sum().to_dict()
        self.item_popularity = {str(k): float(v) for k, v in pop.items()}
        # Compute simple co-occurrence counts per user baskets
        self.cooccurrence = {}
        for user_id, group in df.groupby("user_id"):
            items = group.sort_values("event_strength", ascending=False)["book_id"].tolist()
            for i in range(len(items)):
                a = items[i]
                self.items_set.add(a)
                self.cooccurrence.setdefault(a, {})
                for j in range(i + 1, len(items)):
                    b = items[j]
                    self.items_set.add(b)
                    self.cooccurrence[a][b] = self.cooccurrence[a].get(b, 0) + 1
                    self.cooccurrence.setdefault(b, {})
                    self.cooccurrence[b][a] = self.cooccurrence[b].get(a, 0) + 1

    def _score_by_cooccurrence(self, liked_book_ids: List[str]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for lb in liked_book_ids:
            if lb not in self.cooccurrence:
                continue
            for other, cnt in self.cooccurrence[lb].items():
                scores[other] = scores.get(other, 0.0) + float(cnt)
        # Add popularity prior
        for item, prior in self.item_popularity.items():
            scores[item] = scores.get(item, 0.0) + 0.05 * prior
        return scores

    def score_candidates(self, request: RecommendationRequest, candidate_df: pd.DataFrame) -> Dict[str, float]:
        if candidate_df.empty:
            return {}
        candidate_ids = set(candidate_df["book_id"].astype(str).tolist())
        liked_titles = [t.lower() for t in request.liked_books]
        # Map liked titles to ids using candidate_df
        title_to_id = {str(row["title"]).lower(): str(row["book_id"]) for _, row in candidate_df.iterrows()}
        liked_ids = [title_to_id[t] for t in liked_titles if t in title_to_id]
        if not liked_ids and not self.item_popularity:
            # popularity only from candidate_df
            return {str(row["book_id"]): float(row.get("rating_count", 0.0)) * float(row.get("avg_rating", 0.0)) for _, row in candidate_df.iterrows()}
        co_scores = self._score_by_cooccurrence(liked_ids)
        # Filter to candidates
        filtered = {bid: sc for bid, sc in co_scores.items() if bid in candidate_ids}
        if not filtered:
            # fallback to popularity within candidates
            filtered = {str(row["book_id"]): float(row.get("rating_count", 0.0)) * float(row.get("avg_rating", 0.0)) for _, row in candidate_df.iterrows()}
        return filtered