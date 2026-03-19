import pytest
from httpx import ASGITransport, AsyncClient

from whats_for_dinner.main import app


@pytest.mark.anyio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_recommend_recipe_missing_ingredients():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/recommend_recipe")
    assert response.status_code == 422


@pytest.mark.anyio
@pytest.mark.integration
async def test_recommend_recipe_text():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/recommend_recipe",
            data={"ingredients": "chicken, garlic, soy sauce"},
        )
    assert response.status_code == 200
    body = response.json()
    assert "recipe" in body
    assert len(body["recipe"]) > 0
