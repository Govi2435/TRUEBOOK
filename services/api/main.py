from fastapi import FastAPI
from fastapi import HTTPException
from typing import List, Optional
import os
import yaml

from data_pipeline.schemas import RecommendationRequest, RecommendationResponse, RecommendedBook
from recommender.hybrid import HybridRecommender


def load_config(config_path: str = "config/config.yaml") -> dict:
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


app = FastAPI(title="Global Book Recommender", version="0.1.0")

# Global recommender instance (simple in-memory demo)
CONFIG = load_config()
RECOMMENDER: Optional[HybridRecommender] = None


@app.on_event("startup")
async def startup_event() -> None:
    global RECOMMENDER
    RECOMMENDER = HybridRecommender(CONFIG)
    RECOMMENDER.initialize()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest) -> RecommendationResponse:
    if RECOMMENDER is None:
        raise HTTPException(status_code=503, detail="Recommender not ready")
    try:
        results = RECOMMENDER.recommend(request)
        return RecommendationResponse(recommendations=results)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")