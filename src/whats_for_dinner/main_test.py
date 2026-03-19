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
    monkeypatch.setattr(main, "rag_document_store", object())
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
    monkeypatch.setattr(main, "rag_document_store", object())
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


@pytest.mark.anyio
async def test_recommend_recipe_image_only(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(main, "rag_pipeline", object())
    monkeypatch.setattr(main, "rag_document_store", object())
    async def fake_extract(*_args, **_kwargs) -> str:
        return "tomato, basil"
    monkeypatch.setattr(main, "extract_ingredients_from_upload", fake_extract)
    monkeypatch.setattr(
        main,
        "validate_input_ingredients",
        lambda *_args, **_kwargs: InputValidationResult(
            is_food_only=True,
            non_food_items=[],
            reason="All items are food.",
        ),
    )
    monkeypatch.setattr(main, "recommend_recipe", lambda ingredients, *_args, **_kwargs: f"## {ingredients}")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/recommend_recipe",
            files={"image_file": ("ingredients.jpg", b"fakeimagebytes", "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json() == {"recipe": "## tomato, basil"}


@pytest.mark.anyio
async def test_recommend_recipe_combines_text_and_image(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(main, "rag_pipeline", object())
    monkeypatch.setattr(main, "rag_document_store", object())
    async def fake_extract(*_args, **_kwargs) -> str:
        return "basil"
    monkeypatch.setattr(main, "extract_ingredients_from_upload", fake_extract)
    monkeypatch.setattr(
        main,
        "validate_input_ingredients",
        lambda *_args, **_kwargs: InputValidationResult(
            is_food_only=True,
            non_food_items=[],
            reason="All items are food.",
        ),
    )
    captured: dict[str, str] = {}

    def fake_recommend_recipe(ingredients: str, *_args, **_kwargs) -> str:
        captured["ingredients"] = ingredients
        return "## Test Recipe"

    monkeypatch.setattr(main, "recommend_recipe", fake_recommend_recipe)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/recommend_recipe",
            data={"ingredients": "tomato"},
            files={"image_file": ("ingredients.jpg", b"fakeimagebytes", "image/jpeg")},
        )

    assert response.status_code == 200
    assert captured["ingredients"] == "tomato, basil"
