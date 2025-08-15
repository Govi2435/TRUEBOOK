import pytest
from httpx import AsyncClient, ASGITransport
import asyncio

from services.api.main import app
from scripts.ingest_sample import ensure_sample_data


@pytest.mark.asyncio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_recommend_basic():
    ensure_sample_data()
    # Trigger startup init
    for handler in app.router.on_startup:
        await handler()

    payload = {
        "genres": ["Fantasy"],
        "authors": ["Nnedi Okorafor"],
        "countries": ["Nigeria"],
        "languages": ["en"],
        "themes": ["Afrofuturism"],
        "min_year": 2000,
        "limit": 3,
        "liked_books": ["Akata Witch"]
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/recommend", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "recommendations" in data
        recs = data["recommendations"]
        assert isinstance(recs, list)
        assert len(recs) > 0
        assert all("title" in r and "explanation" in r for r in recs)