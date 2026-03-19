import pytest
from httpx import ASGITransport, AsyncClient

from whats_for_dinner import main
from whats_for_dinner.main import app
from whats_for_dinner.models import InputValidationResult


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


@pytest.mark.anyio
async def test_recommend_recipe_rejects_non_food(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(main, "rag_pipeline", object())
    monkeypatch.setattr(
        main,
        "validate_input_ingredients",
        lambda *_args, **_kwargs: InputValidationResult(
            is_food_only=False,
            non_food_items=["soap"],
            reason="Detected non-food item.",
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/recommend_recipe",
            data={"ingredients": "chicken, soap"},
        )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["message"] == "Ingredients must be edible food items only."
    assert "soap" in detail["non_food_items"]


@pytest.mark.anyio
async def test_recommend_recipe_allows_food_only(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(main, "rag_pipeline", object())
    monkeypatch.setattr(
        main,
        "validate_input_ingredients",
        lambda *_args, **_kwargs: InputValidationResult(
            is_food_only=True,
            non_food_items=[],
            reason="All items are food.",
        ),
    )
    monkeypatch.setattr(main, "recommend_recipe", lambda *_args, **_kwargs: "## Test Recipe")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/recommend_recipe",
            data={"ingredients": "chicken, garlic"},
        )

    assert response.status_code == 200
    assert response.json() == {"recipe": "## Test Recipe"}
