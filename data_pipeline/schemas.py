from pydantic import BaseModel, Field
from typing import List, Optional


class RecommendationRequest(BaseModel):
    genres: List[str] = Field(default_factory=list)
    authors: List[str] = Field(default_factory=list)
    countries: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    liked_books: List[str] = Field(default_factory=list)
    limit: int = 10


class RecommendedBook(BaseModel):
    book_id: str
    title: str
    author: str
    country: Optional[str] = None
    language: Optional[str] = None
    genres: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    score: float = 0.0
    explanation: str = ""


class RecommendationResponse(BaseModel):
    recommendations: List[RecommendedBook] = Field(default_factory=list)