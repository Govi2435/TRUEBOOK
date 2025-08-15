from __future__ import annotations

from typing import List, Optional, Tuple
import os
import math
import numpy as np
import pandas as pd

from data_pipeline.schemas import RecommendationRequest, RecommendedBook
from recommender.content_based import ContentBasedRecommender
from recommender.collaborative import CollaborativeRecommender


class HybridRecommender:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.alpha = float(self.config.get("recommendation", {}).get("hybrid_alpha", 0.6))
        self.diversity_weight = float(self.config.get("recommendation", {}).get("diversity_weight", 0.15))
        paths = self.config.get("paths", {})
        self.books_csv = paths.get("books_csv", "sample_data/books_sample.csv")
        self.interactions_csv = paths.get("interactions_csv", "sample_data/user_interactions_sample.csv")

        self.books: pd.DataFrame | None = None
        self.content_model: ContentBasedRecommender | None = None
        self.collab_model: CollaborativeRecommender | None = None

    def initialize(self) -> None:
        self._load_books()
        self.content_model = ContentBasedRecommender(self.config)
        self.content_model.fit(self.books)
        self.collab_model = CollaborativeRecommender(self.config)
        self.collab_model.fit(self.interactions_csv, self.books)

    def _load_books(self) -> None:
        if not os.path.exists(self.books_csv):
            raise ValueError(f"Books CSV not found: {self.books_csv}")
        self.books = pd.read_csv(self.books_csv)
        # Normalize columns
        if "genres" in self.books.columns:
            self.books["genres"] = self.books["genres"].fillna("")
        if "themes" in self.books.columns:
            self.books["themes"] = self.books["themes"].fillna("")
        for col in ["author", "country", "language", "title"]:
            if col in self.books.columns:
                self.books[col] = self.books[col].fillna("")

    def _apply_filters(self, df: pd.DataFrame, request: RecommendationRequest) -> pd.DataFrame:
        filtered = df
        if request.genres:
            filtered = filtered[filtered["genres"].str.contains("|".join(map(lambda g: rf"\b{g}\b", request.genres)), case=False, regex=True)]
        if request.authors:
            filtered = filtered[filtered["author"].str.lower().isin([a.lower() for a in request.authors])]
        if request.countries:
            filtered = filtered[filtered["country"].str.lower().isin([c.lower() for c in request.countries])]
        if request.languages:
            filtered = filtered[filtered["language"].str.lower().isin([l.lower() for l in request.languages])]
        if request.min_year is not None:
            filtered = filtered[filtered["year"] >= request.min_year]
        if request.max_year is not None:
            filtered = filtered[filtered["year"] <= request.max_year]
        return filtered

    def _blend_scores(self, content_scores: dict[str, float], collab_scores: dict[str, float]) -> dict[str, float]:
        all_ids = set(content_scores) | set(collab_scores)
        blended: dict[str, float] = {}
        for book_id in all_ids:
            cs = content_scores.get(book_id, 0.0)
            ms = collab_scores.get(book_id, 0.0)
            blended[book_id] = self.alpha * cs + (1.0 - self.alpha) * ms
        return blended

    def _apply_diversity_boost(self, df: pd.DataFrame, scores: dict[str, float]) -> dict[str, float]:
        if df.empty or not scores:
            return scores
        # Simple diversity term: penalize over-represented countries/languages among top candidates
        diversity_weight = self.diversity_weight
        counts_country = df["country"].str.lower().value_counts(dropna=False).to_dict()
        counts_language = df["language"].str.lower().value_counts(dropna=False).to_dict()
        max_cc = max(counts_country.values()) if counts_country else 1
        max_cl = max(counts_language.values()) if counts_language else 1
        adjusted = {}
        for book_id, base in scores.items():
            row = df[df["book_id"].astype(str) == str(book_id)]
            if row.empty:
                adjusted[book_id] = base
                continue
            country = str(row.iloc[0]["country"]).lower()
            language = str(row.iloc[0]["language"]).lower()
            rarity = 0.0
            if country:
                rarity += 1.0 - (counts_country.get(country, 0) / max_cc)
            if language:
                rarity += 1.0 - (counts_language.get(language, 0) / max_cl)
            adjusted[book_id] = base * (1.0 + diversity_weight * (rarity / 2.0))
        return adjusted

    def _build_explanation(self, row: pd.Series, request: RecommendationRequest, source_notes: List[str]) -> str:
        reasons: List[str] = []
        if request.genres and any(g.lower() in str(row.get("genres", "")).lower() for g in request.genres):
            reasons.append("matches your preferred genre")
        if request.themes and any(t.lower() in str(row.get("themes", "")).lower() for t in request.themes):
            reasons.append("aligns with your themes")
        if request.authors and str(row.get("author", "")).lower() in [a.lower() for a in request.authors]:
            reasons.append("by your preferred author")
        if request.countries and str(row.get("country", "")).lower() in [c.lower() for c in request.countries]:
            reasons.append("from your selected country")
        if request.languages and str(row.get("language", "")).lower() in [l.lower() for l in request.languages]:
            reasons.append("in your preferred language")
        if request.min_year or request.max_year:
            reasons.append("within your publication year range")
        explanation = "; ".join(reasons[:2]) if reasons else "personalized based on your preferences"
        if source_notes:
            explanation += f"; signal: {', '.join(source_notes)}"
        return explanation

    def recommend(self, request: RecommendationRequest) -> List[RecommendedBook]:
        if self.books is None:
            raise ValueError("Books not loaded")

        candidate_df = self._apply_filters(self.books.copy(), request)
        if candidate_df.empty:
            # Fallback to popularity among all books
            candidate_df = self.books.copy()

        content_scores = self.content_model.score_candidates(request, candidate_df)
        collab_scores = self.collab_model.score_candidates(request, candidate_df)

        blended = self._blend_scores(content_scores, collab_scores)
        blended = self._apply_diversity_boost(candidate_df, blended)

        # Rank and build response
        ranked = sorted(blended.items(), key=lambda kv: kv[1], reverse=True)
        results: List[RecommendedBook] = []
        used_ids: set[str] = set([str(t).lower() for t in request.liked_books])
        for book_id, score in ranked:
            if len(results) >= max(1, request.limit):
                break
            row = candidate_df[candidate_df["book_id"].astype(str) == str(book_id)]
            if row.empty:
                continue
            title_lower = str(row.iloc[0]["title"]).lower()
            if title_lower in used_ids:
                continue
            source_notes = []
            if book_id in content_scores:
                source_notes.append("content")
            if book_id in collab_scores:
                source_notes.append("collab")
            explanation = self._build_explanation(row.iloc[0], request, source_notes)
            results.append(
                RecommendedBook(
                    book_id=str(row.iloc[0]["book_id"]),
                    title=str(row.iloc[0]["title"]),
                    author=str(row.iloc[0]["author"]),
                    country=str(row.iloc[0].get("country", "")) or None,
                    language=str(row.iloc[0].get("language", "")) or None,
                    genres=[g.strip() for g in str(row.iloc[0].get("genres", "")).split("|") if g.strip()],
                    year=int(row.iloc[0]["year"]) if not pd.isna(row.iloc[0]["year"]) else None,
                    score=float(score),
                    explanation=explanation,
                )
            )
        return results